"""
Protocol Agent - Agente ADK especializado en protocolos y conocimiento médico
Implementa Agent Development Kit (ADK) para sistemas de conocimiento médico.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.agents.base_agent import BaseAgent, AgentCapability, AgentMessage, AgentResponse
from src.systems.medical_knowledge import MedicalKnowledgeSystem, MedicalQuery, QueryType
from src.redis_layer.vector_service import VectorService
from src.redis_layer.protocol_indexer import ProtocolIndexer
from src.utils.secure_logger import SecureLogger

logger = SecureLogger("protocol_agent")


class ProtocolQueryType(Enum):
    """Tipos específicos de consulta de protocolos."""
    CLINICAL_PROTOCOL = "clinical_protocol"
    TREATMENT_GUIDELINE = "treatment_guideline"
    MEDICATION_PROTOCOL = "medication_protocol"
    PREVENTION_PROTOCOL = "prevention_protocol"
    EMERGENCY_PROTOCOL = "emergency_protocol"
    INSTITUTIONAL_POLICY = "institutional_policy"
    BEST_PRACTICES = "best_practices"


@dataclass
class ProtocolRequest:
    """Solicitud de protocolo médico."""
    protocol_type: ProtocolQueryType
    clinical_context: str
    patient_context: Optional[Dict[str, Any]] = None
    urgency_level: str = "routine"
    language: str = "es"
    include_references: bool = True
    include_evidence_level: bool = True


@dataclass
class ProtocolResult:
    """Resultado de búsqueda de protocolo."""
    protocol_found: bool
    protocol_content: Dict[str, Any]
    evidence_level: str
    confidence_score: float
    references: List[str]
    last_updated: Optional[str]
    validation_status: str
    clinical_recommendations: List[str]
    follow_up_actions: List[str]


class ProtocolAgent(BaseAgent):
    """
    Agent especializado en protocolos y conocimiento médico.
    
    Responsabilidades:
    - Búsqueda de protocolos clínicos validados
    - Consultas de conocimiento médico estructurado
    - Recomendaciones basadas en evidencia
    - Integración con sistemas de conocimiento
    """
    
    def __init__(self, 
                 agent_id: str = "protocol_agent",
                 medical_knowledge_system: Optional[MedicalKnowledgeSystem] = None):
        """
        Inicializar ProtocolAgent.
        
        Args:
            agent_id: Identificador único del agente
            medical_knowledge_system: Sistema de conocimiento médico
        """
        super().__init__(
            agent_id=agent_id,
            name="Protocol Agent",
            description="Agente especializado en protocolos y conocimiento médico",
            capabilities=[
                AgentCapability.MEDICAL_PROTOCOL_SEARCH,
                AgentCapability.CLINICAL_KNOWLEDGE,
                AgentCapability.EVIDENCE_RETRIEVAL,
                AgentCapability.MEDICAL_RECOMMENDATIONS
            ],
            version="1.0.0"
        )
        
        # Sistema de conocimiento médico
        self.medical_system = medical_knowledge_system or MedicalKnowledgeSystem()
        
        # Configuración específica del agente
        self.protocol_config = {
            "max_protocols_per_query": 5,
            "min_confidence_threshold": 0.6,
            "require_evidence": True,
            "include_disclaimers": True,
            "max_response_time": 30.0  # segundos
        }
        
        # Registro de auditoría
        logger.audit("protocol_agent_initialized", {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "medical_system": "active",
            "config": self.protocol_config
        })
    
    async def initialize(self) -> bool:
        """
        Inicializar el agente y sus dependencias.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Inicializar sistema médico
            await self.medical_system.initialize()
            
            # Marcar como inicializado
            self.is_initialized = True
            
            logger.audit("protocol_agent_ready", {
                "agent_id": self.agent_id,
                "initialization_successful": True,
                "medical_system_ready": True
            })
            
            return True
            
        except Exception as e:
            logger.error("protocol_agent_initialization_failed", {
                "agent_id": self.agent_id,
                "error": str(e)
            })
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Procesar mensaje y generar respuesta de protocolo.
        
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
                    "Invalid message format for protocol query"
                )
            
            # Extraer solicitud de protocolo
            protocol_request = self._extract_protocol_request(message)
            
            # Buscar protocolo
            protocol_result = await self._search_protocol(protocol_request)
            
            # Generar respuesta
            response = await self._generate_protocol_response(
                message,
                protocol_request, 
                protocol_result,
                start_time
            )
            
            # Registrar procesamiento exitoso
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.audit("protocol_query_processed", {
                "agent_id": self.agent_id,
                "session_id": message.session_id,
                "protocol_type": protocol_request.protocol_type.value,
                "protocol_found": protocol_result.protocol_found,
                "confidence": protocol_result.confidence_score,
                "processing_time": processing_time
            })
            
            return response
            
        except Exception as e:
            logger.error("protocol_processing_failed", {
                "agent_id": self.agent_id,
                "session_id": message.session_id,
                "error": str(e)
            })
            
            return self._create_error_response(message, str(e))
    
    async def _search_protocol(self, request: ProtocolRequest) -> ProtocolResult:
        """
        Buscar protocolo médico.
        
        Args:
            request: Solicitud de protocolo
            
        Returns:
            ProtocolResult: Resultado de la búsqueda
        """
        try:
            # Crear consulta médica
            medical_query = MedicalQuery(
                session_id=f"protocol_{datetime.now().timestamp()}",
                query_text=request.clinical_context,
                query_type=self._convert_protocol_type(request.protocol_type),
                context=request.clinical_context,
                patient_context=json.dumps(request.patient_context) if request.patient_context else None,
                urgency=request.urgency_level,
                language=request.language
            )
            
            # Procesar con sistema médico (simulamos triage_decision)
            from ..core.medical_dispatcher import TriageDecision
            mock_triage = TriageDecision(
                action="knowledge_search",
                priority="medium",
                destination="medical_knowledge",
                reason="Protocol search request",
                confidence=0.9,
                flags=[],
                session_timeout=900
            )
            
            # Usar StandardizedInput mock
            from ..core.input_packager import StandardizedInput
            mock_input = StandardizedInput(
                session_id=medical_query.session_id,
                input_type="text",
                content={"text": request.clinical_context},
                raw_content={"text": request.clinical_context},
                metadata={
                    "source": "protocol_agent",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                security_context={
                    "encryption_enabled": True,
                    "phi_detected": False
                }
            )
            
            # Procesar con sistema médico
            result = await self.medical_system.process(mock_input, mock_triage)
            
            if result["success"]:
                medical_response = result["medical_response"]
                
                return ProtocolResult(
                    protocol_found=True,
                    protocol_content=medical_response.get("knowledge_result", {}),
                    evidence_level=medical_response.get("evidence_level", "moderate"),
                    confidence_score=medical_response.get("knowledge_result", {}).get("confidence", 0.0),
                    references=medical_response.get("knowledge_result", {}).get("references", []),
                    last_updated=medical_response.get("timestamp"),
                    validation_status="validated",
                    clinical_recommendations=medical_response.get("clinical_recommendations", []),
                    follow_up_actions=medical_response.get("follow_up_suggestions", [])
                )
            else:
                return ProtocolResult(
                    protocol_found=False,
                    protocol_content={},
                    evidence_level="N/A",
                    confidence_score=0.0,
                    references=[],
                    last_updated=None,
                    validation_status="not_found",
                    clinical_recommendations=[],
                    follow_up_actions=["Consultar con especialista médico"]
                )
                
        except Exception as e:
            logger.error("protocol_search_failed", {
                "request": asdict(request),
                "error": str(e)
            })
            
            return ProtocolResult(
                protocol_found=False,
                protocol_content={"error": str(e)},
                evidence_level="N/A",
                confidence_score=0.0,
                references=[],
                last_updated=None,
                validation_status="error",
                clinical_recommendations=[],
                follow_up_actions=["Reintentar búsqueda", "Escalación manual"]
            )
    
    def _extract_protocol_request(self, message: AgentMessage) -> ProtocolRequest:
        """
        Extraer solicitud de protocolo del mensaje.
        
        Args:
            message: Mensaje del agente
            
        Returns:
            ProtocolRequest: Solicitud de protocolo
        """
        content = message.content
        
        # Determinar tipo de protocolo
        protocol_type = self._classify_protocol_type(content.get("text", ""))
        
        # Extraer contexto clínico
        clinical_context = content.get("clinical_context", content.get("text", ""))
        
        # Extraer contexto del paciente
        patient_context = content.get("patient_context")
        
        # Determinar urgencia
        urgency = content.get("urgency", "routine")
        
        # Detectar idioma
        language = content.get("language", "es")
        
        return ProtocolRequest(
            protocol_type=protocol_type,
            clinical_context=clinical_context,
            patient_context=patient_context,
            urgency_level=urgency,
            language=language,
            include_references=content.get("include_references", True),
            include_evidence_level=content.get("include_evidence_level", True)
        )
    
    def _classify_protocol_type(self, text: str) -> ProtocolQueryType:
        """
        Clasificar tipo de protocolo basado en el texto.
        
        Args:
            text: Texto a clasificar
            
        Returns:
            ProtocolQueryType: Tipo de protocolo
        """
        text_lower = text.lower()
        
        # Palabras clave por tipo
        type_keywords = {
            ProtocolQueryType.CLINICAL_PROTOCOL: [
                "protocolo clínico", "clinical protocol", "procedimiento médico",
                "guía clínica", "clinical guideline"
            ],
            ProtocolQueryType.TREATMENT_GUIDELINE: [
                "tratamiento", "treatment", "terapia", "therapy",
                "manejo", "management"
            ],
            ProtocolQueryType.MEDICATION_PROTOCOL: [
                "medicamento", "medication", "fármaco", "drug",
                "prescripción", "prescription"
            ],
            ProtocolQueryType.PREVENTION_PROTOCOL: [
                "prevención", "prevention", "profilaxis", "prophylaxis",
                "medidas preventivas"
            ],
            ProtocolQueryType.EMERGENCY_PROTOCOL: [
                "emergencia", "emergency", "urgencia", "urgent",
                "crítico", "critical"
            ],
            ProtocolQueryType.INSTITUTIONAL_POLICY: [
                "política", "policy", "normativa", "regulation",
                "institucional", "institutional"
            ]
        }
        
        # Buscar coincidencias
        for protocol_type, keywords in type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return protocol_type
        
        # Default
        return ProtocolQueryType.CLINICAL_PROTOCOL
    
    def _convert_protocol_type(self, protocol_type: ProtocolQueryType) -> QueryType:
        """
        Convertir tipo de protocolo a tipo de consulta médica.
        
        Args:
            protocol_type: Tipo de protocolo
            
        Returns:
            QueryType: Tipo de consulta médica
        """
        conversion_map = {
            ProtocolQueryType.CLINICAL_PROTOCOL: QueryType.PROTOCOL_SEARCH,
            ProtocolQueryType.TREATMENT_GUIDELINE: QueryType.TREATMENT_RECOMMENDATION,
            ProtocolQueryType.MEDICATION_PROTOCOL: QueryType.MEDICATION_INFO,
            ProtocolQueryType.PREVENTION_PROTOCOL: QueryType.PREVENTION_PROTOCOL,
            ProtocolQueryType.EMERGENCY_PROTOCOL: QueryType.DIAGNOSTIC_SUPPORT,
            ProtocolQueryType.INSTITUTIONAL_POLICY: QueryType.CLINICAL_GUIDELINE,
            ProtocolQueryType.BEST_PRACTICES: QueryType.CLINICAL_GUIDELINE
        }
        
        return conversion_map.get(protocol_type, QueryType.PROTOCOL_SEARCH)
    
    async def _generate_protocol_response(self, 
                                        message: AgentMessage,
                                        request: ProtocolRequest,
                                        result: ProtocolResult,
                                        start_time: datetime) -> AgentResponse:
        """
        Generar respuesta de protocolo.
        
        Args:
            message: Mensaje original
            request: Solicitud de protocolo
            result: Resultado de búsqueda
            start_time: Tiempo de inicio
            
        Returns:
            AgentResponse: Respuesta del agente
        """
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        if result.protocol_found:
            # Protocolo encontrado
            response_content = {
                "protocol_result": asdict(result),
                "status": "protocol_found",
                "confidence": result.confidence_score,
                "evidence_level": result.evidence_level,
                "clinical_recommendations": result.clinical_recommendations,
                "follow_up_actions": result.follow_up_actions,
                "disclaimers": [
                    "Esta información es solo para fines educativos",
                    "No reemplaza el criterio médico profesional",
                    "Consulte siempre con un profesional de la salud"
                ]
            }
            
            success = True
            response_text = f"Protocolo encontrado: {result.evidence_level} nivel de evidencia"
            
        else:
            # Protocolo no encontrado
            response_content = {
                "protocol_result": asdict(result),
                "status": "protocol_not_found",
                "reason": "No se encontró protocolo específico",
                "suggestions": result.follow_up_actions,
                "disclaimers": [
                    "Consultar con especialista médico",
                    "Revisar protocolos institucionales"
                ]
            }
            
            success = False
            response_text = "Protocolo no encontrado - se requiere consulta médica"
        
        return AgentResponse(
            session_id=message.session_id,
            agent_id=self.agent_id,
            success=success,
            content=response_content,
            metadata={
                "processing_time": processing_time,
                "protocol_type": request.protocol_type.value,
                "confidence": result.confidence_score,
                "evidence_level": result.evidence_level,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            message=response_text,
            next_actions=result.follow_up_actions,
            requires_human_review=not result.protocol_found or result.confidence_score < 0.7
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
        
        # Debe tener texto o contexto clínico
        text = message.content.get("text", "")
        clinical_context = message.content.get("clinical_context", "")
        
        if not text and not clinical_context:
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
            session_id=message.session_id,
            agent_id=self.agent_id,
            success=False,
            content={
                "error": error,
                "status": "error",
                "disclaimers": [
                    "Error en procesamiento de protocolo",
                    "Contactar soporte técnico si persiste"
                ]
            },
            metadata={
                "error_occurred": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            message=f"Error procesando protocolo: {error}",
            next_actions=["Reintentar consulta", "Contactar soporte"],
            requires_human_review=True
        )
    
    async def shutdown(self):
        """Cerrar agente y liberar recursos."""
        try:
            logger.audit("protocol_agent_shutdown", {
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Realizar limpieza si es necesario
            self.is_initialized = False
            
        except Exception as e:
            logger.error("protocol_agent_shutdown_error", {
                "agent_id": self.agent_id,
                "error": str(e)
            })


# Factory para crear ProtocolAgent
class ProtocolAgentFactory:
    """Factory para crear instancias de ProtocolAgent."""
    
    @staticmethod
    async def create_agent(config: Optional[Dict[str, Any]] = None) -> ProtocolAgent:
        """
        Crear instancia de ProtocolAgent.
        
        Args:
            config: Configuración opcional
            
        Returns:
            ProtocolAgent: Instancia configurada
        """
        config = config or {}
        
        # Crear sistema de conocimiento médico si se especifica
        medical_system = None
        if config.get("use_medical_system", True):
            medical_system = MedicalKnowledgeSystem()
        
        # Crear agente
        agent = ProtocolAgent(
            agent_id=config.get("agent_id", "protocol_agent"),
            medical_knowledge_system=medical_system
        )
        
        # Inicializar
        await agent.initialize()
        
        return agent