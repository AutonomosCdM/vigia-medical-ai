"""
Medical Dispatcher - Capa 2: Orquestador Médico
Decide la ruta de procesamiento basada en el contenido médico.

Responsabilidades:
- Determinar ruta de procesamiento (triage)
- Validar permisos por capa
- Orquestar flujo entre sistemas especializados
- Mantener audit trail médico
- Aplicar timeouts médicos críticos
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import json

from .input_packager import StandardizedInput
from .session_manager import SessionManager, SessionState, SessionType
from .phi_tokenization_client import PHITokenizationClient, TokenizedPatient
from ..utils.secure_logger import SecureLogger
from ..utils.validators import validate_patient_code_format

logger = SecureLogger("medical_dispatcher")


class ProcessingRoute(Enum):
    """Rutas de procesamiento disponibles."""
    CLINICAL_IMAGE = "clinical_image_processing"
    MEDICAL_QUERY = "medical_knowledge_system"
    HUMAN_REVIEW = "human_review_queue"
    EMERGENCY = "emergency_escalation"
    INVALID = "invalid_input"
    
    # FASE 2 - Multimodal Analysis Routes
    VOICE_ANALYSIS_REQUIRED = "voice_analysis_required"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"  # Voice + Image
    FASE2_COMPLETE = "fase2_complete"


class TriageDecision:
    """Decisión de triage con justificación."""
    def __init__(self, route: ProcessingRoute, confidence: float, reason: str, 
                 flags: Optional[List[str]] = None):
        self.route = route
        self.confidence = confidence
        self.reason = reason
        self.flags = flags or []
        self.timestamp = datetime.now(timezone.utc)
        self.tokenized_patient: Optional[TokenizedPatient] = None  # NO PHI data
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "route": self.route.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "flags": self.flags,
            "timestamp": self.timestamp.isoformat()
        }
        
        # Include tokenized patient data if available (NO PHI)
        if self.tokenized_patient:
            result["tokenized_patient"] = self.tokenized_patient.to_dict()
        
        return result


class MedicalDispatcher:
    """
    Orquestador médico principal.
    Coordina el flujo entre sistemas especializados.
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None, 
                 phi_client: Optional[PHITokenizationClient] = None):
        """
        Inicializar dispatcher.
        
        Args:
            session_manager: Gestor de sesiones (se crea uno si no se proporciona)
            phi_client: Cliente PHI tokenization (se crea uno si no se proporciona)
        """
        self.session_manager = session_manager or SessionManager()
        self.phi_client = phi_client  # Se inicializa en initialize()
        
        # Configuración de timeouts por ruta
        self.route_timeouts = {
            ProcessingRoute.CLINICAL_IMAGE: 30,  # segundos
            ProcessingRoute.MEDICAL_QUERY: 20,
            ProcessingRoute.HUMAN_REVIEW: 300,  # 5 minutos
            ProcessingRoute.EMERGENCY: 60,
            ProcessingRoute.INVALID: 5
        }
        
        # Procesadores registrados por ruta
        self.route_processors = {}
        
        # Métricas de routing
        self.routing_metrics = {
            "total_dispatched": 0,
            "routes": {route.value: 0 for route in ProcessingRoute}
        }
        
        logger.audit("medical_dispatcher_initialized", {
            "component": "layer2_dispatcher",
            "routes_available": [r.value for r in ProcessingRoute],
            "medical_compliance": True,
            "phi_tokenization_enabled": True
        })
    
    async def initialize(self):
        """Inicializar dispatcher y servicios."""
        await self.session_manager.initialize()
        
        # Inicializar PHI tokenization client si no se proporcionó
        if self.phi_client is None:
            self.phi_client = PHITokenizationClient()
            await self.phi_client.initialize()
        
        logger.audit("medical_dispatcher_ready", {
            "status": "initialized",
            "session_manager": "active",
            "phi_tokenization": "ready"
        })
    
    async def dispatch(self, standardized_input: StandardizedInput) -> Dict[str, Any]:
        """
        Despachar input estandarizado a la ruta apropiada.
        
        Args:
            standardized_input: Input del Input Queue
            
        Returns:
            Dict con resultado del procesamiento
        """
        session_id = standardized_input.session_id
        
        try:
            # Realizar triage médico
            triage_decision = await self._perform_medical_triage(standardized_input)
            
            # Log de decisión de triage (sin PII)
            logger.audit("medical_triage_decision", {
                "session_id": session_id,
                "route": triage_decision.route.value,
                "confidence": triage_decision.confidence,
                "reason": triage_decision.reason,
                "flags": triage_decision.flags
            })
            
            # Determinar tipo de sesión basado en ruta
            session_type = self._route_to_session_type(triage_decision.route)
            is_emergency = "emergency" in triage_decision.flags
            
            # Crear sesión médica
            session_result = await self.session_manager.create_session(
                input_data={
                    "source": standardized_input.metadata.get("source"),
                    "input_type": standardized_input.input_type
                },
                session_type=session_type,
                emergency=is_emergency
            )
            
            if not session_result["success"]:
                return {
                    "success": False,
                    "error": "session_creation_failed",
                    "details": session_result.get("error"),
                    "session_id": session_id
                }
            
            # Actualizar estado a triaging
            await self.session_manager.update_session_state(
                session_id, 
                SessionState.TRIAGING,
                additional_data=triage_decision.to_dict()
            )
            
            # Despachar a ruta específica
            dispatch_result = await self._dispatch_to_route(
                standardized_input,
                triage_decision,
                session_result["timeout_minutes"]
            )
            
            # Actualizar métricas
            self.routing_metrics["total_dispatched"] += 1
            self.routing_metrics["routes"][triage_decision.route.value] += 1
            
            return {
                "success": True,
                "session_id": session_id,
                "route": triage_decision.route.value,
                "processing_result": dispatch_result,
                "triage_confidence": triage_decision.confidence,
                "session_expires": session_result["expires_at"]
            }
            
        except Exception as e:
            logger.error("medical_dispatch_failed", {
                "session_id": session_id,
                "error": str(e),
                "input_type": standardized_input.input_type
            })
            
            # Marcar sesión como fallida
            await self.session_manager.update_session_state(
                session_id,
                SessionState.FAILED,
                additional_data={"error": str(e)}
            )
            
            return {
                "success": False,
                "error": "dispatch_failed",
                "details": str(e),
                "session_id": session_id
            }
    
    async def _perform_medical_triage(self, standardized_input: StandardizedInput) -> TriageDecision:
        """
        Realizar triage médico del input.
        
        CRITICAL: Este es el punto de decisión médica principal
        """
        try:
            raw_content = standardized_input.raw_content
            metadata = standardized_input.metadata
            
            # Extraer indicadores clave
            has_image = metadata.get("has_media", False)
            has_text = metadata.get("has_text", False)
            has_voice = metadata.get("has_voice", False) or metadata.get("has_audio", False)
            text_content = raw_content.get("text", "").strip()
            
            # Detectar si es una consulta multimodal (FASE 2)
            is_multimodal_context = self._is_multimodal_medical_context(text_content)
            
            # Buscar indicadores de emergencia
            emergency_keywords = [
                "urgente", "emergencia", "dolor severo", "sangrado",
                "urgent", "emergency", "severe pain", "bleeding",
                "crítico", "critical", "ayuda", "help"
            ]
            
            is_emergency = any(
                keyword in text_content.lower() 
                for keyword in emergency_keywords
            )
            
            # Detectar y tokenizar código de paciente (Bruce Wayne → Batman)
            patient_code = self._extract_patient_code(text_content)
            tokenized_patient = None
            
            if patient_code:
                try:
                    # Convertir MRN a tokenized patient (PHI → NO PHI)
                    tokenized_patient = await self._tokenize_patient_data(patient_code)
                    logger.audit("patient_tokenization_success", {
                        "original_code": patient_code[:8] + "...",  # Partial for audit
                        "patient_alias": tokenized_patient.patient_alias,
                        "token_id": tokenized_patient.token_id
                    })
                except Exception as e:
                    logger.error("patient_tokenization_failed", {
                        "error": str(e),
                        "patient_code": patient_code[:8] + "..."
                    })
                    tokenized_patient = None
            
            has_valid_patient_code = tokenized_patient is not None
            
            # Lógica de triage basada en el documento de arquitectura
            
            # Ruta 1: Análisis multimodal (FASE 2) - Imagen + Voz
            if has_image and has_voice and has_valid_patient_code and is_multimodal_context:
                decision = TriageDecision(
                    route=ProcessingRoute.MULTIMODAL_ANALYSIS,
                    confidence=0.98,
                    reason="Multimodal analysis required: Image + Voice data with valid tokenized patient",
                    flags=["has_tokenized_patient", "multimodal_context", "phi_protected", "fase2_trigger"]
                )
                decision.tokenized_patient = tokenized_patient
                decision.analysis_mode = "multimodal"
                return decision
            
            # Ruta 1B: Imagen con posible voz requerida (FASE 2 parcial)
            if has_image and has_valid_patient_code and is_multimodal_context and not has_voice:
                decision = TriageDecision(
                    route=ProcessingRoute.VOICE_ANALYSIS_REQUIRED,
                    confidence=0.90,
                    reason="Image analysis complete, voice analysis required for FASE 2",
                    flags=["has_tokenized_patient", "voice_required", "phi_protected", "fase2_pending"]
                )
                decision.tokenized_patient = tokenized_patient
                decision.analysis_mode = "voice_pending"
                return decision
            
            # Ruta 1C: Imagen clínica estándar (FASE 1)
            if has_image and has_valid_patient_code:
                decision = TriageDecision(
                    route=ProcessingRoute.CLINICAL_IMAGE,
                    confidence=0.95,
                    reason="Clinical image with valid tokenized patient",
                    flags=["has_tokenized_patient", "clinical_context", "phi_protected"]
                )
                # Adjuntar datos tokenizados (NO PHI)
                decision.tokenized_patient = tokenized_patient
                decision.analysis_mode = "image_only"
                return decision
            
            # Ruta 2: Consulta médica estructurada
            if has_text and not has_image:
                # Detectar consulta médica vs texto aleatorio
                medical_indicators = [
                    "protocolo", "protocol", "tratamiento", "treatment",
                    "diagnóstico", "diagnosis", "síntomas", "symptoms",
                    "medicamento", "medication", "dosis", "dose",
                    "lesión", "injury", "lpp", "úlcera", "ulcer"
                ]
                
                is_medical_query = any(
                    indicator in text_content.lower()
                    for indicator in medical_indicators
                )
                
                if is_medical_query:
                    return TriageDecision(
                        route=ProcessingRoute.MEDICAL_QUERY,
                        confidence=0.85,
                        reason="Medical knowledge query detected",
                        flags=["medical_query", "text_only"]
                    )
            
            # Ruta 3: Escalamiento de emergencia
            if is_emergency:
                return TriageDecision(
                    route=ProcessingRoute.EMERGENCY,
                    confidence=0.90,
                    reason="Emergency keywords detected",
                    flags=["emergency", "requires_human_review"]
                )
            
            # Ruta 4: Casos ambiguos - revisión humana
            if has_image and not has_valid_patient_code:
                return TriageDecision(
                    route=ProcessingRoute.HUMAN_REVIEW,
                    confidence=0.70,
                    reason="Image without patient code - requires human validation",
                    flags=["missing_patient_code", "ambiguous_context"]
                )
            
            # Ruta 5: Input inválido
            return TriageDecision(
                route=ProcessingRoute.INVALID,
                confidence=0.95,
                reason="No valid medical context detected",
                flags=["invalid_input", "no_medical_content"]
            )
            
        except Exception as e:
            logger.error("medical_triage_failed", {
                "error": str(e),
                "session_id": standardized_input.session_id
            })
            
            # En caso de error, enviar a revisión humana
            return TriageDecision(
                route=ProcessingRoute.HUMAN_REVIEW,
                confidence=0.50,
                reason=f"Triage error: {str(e)}",
                flags=["triage_error", "requires_human_review"]
            )
    
    async def _dispatch_to_route(self, 
                                standardized_input: StandardizedInput,
                                triage_decision: TriageDecision,
                                timeout_minutes: int) -> Dict[str, Any]:
        """
        Despachar a la ruta específica de procesamiento.
        
        Args:
            standardized_input: Input estandarizado
            triage_decision: Decisión de triage
            timeout_minutes: Timeout de sesión en minutos
            
        Returns:
            Dict con resultado del procesamiento
        """
        session_id = standardized_input.session_id
        route = triage_decision.route
        
        try:
            # Actualizar estado a procesando
            await self.session_manager.update_session_state(
                session_id,
                SessionState.PROCESSING,
                processor_id=route.value
            )
            
            # Obtener timeout para la ruta
            timeout_seconds = self.route_timeouts.get(route, 30)
            
            # Verificar si hay procesador registrado
            if route in self.route_processors:
                processor = self.route_processors[route]
                
                # Ejecutar con timeout
                try:
                    result = await asyncio.wait_for(
                        processor.process(standardized_input, triage_decision),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise Exception(f"Processing timeout after {timeout_seconds} seconds")
            else:
                # Simulación de procesamiento por ruta
                result = await self._simulate_route_processing(
                    standardized_input, 
                    triage_decision
                )
            
            # Marcar como completado
            await self.session_manager.update_session_state(
                session_id,
                SessionState.COMPLETED,
                additional_data={"result_summary": result.get("summary", "completed")}
            )
            
            return result
            
        except Exception as e:
            logger.error("route_dispatch_failed", {
                "session_id": session_id,
                "route": route.value,
                "error": str(e)
            })
            
            # Marcar como fallido
            await self.session_manager.update_session_state(
                session_id,
                SessionState.FAILED,
                additional_data={"error": str(e), "route": route.value}
            )
            
            raise
    
    async def _simulate_route_processing(self,
                                       standardized_input: StandardizedInput,
                                       triage_decision: TriageDecision) -> Dict[str, Any]:
        """
        Simular procesamiento por ruta (para testing).
        En producción, esto sería reemplazado por procesadores reales.
        """
        route = triage_decision.route
        session_id = standardized_input.session_id
        
        # Simular delay de procesamiento
        await asyncio.sleep(2)
        
        if route == ProcessingRoute.CLINICAL_IMAGE:
            return {
                "success": True,
                "route": route.value,
                "detection_result": {
                    "lpp_detected": True,
                    "lpp_grade": 2,
                    "confidence": 0.87,
                    "bounding_boxes": [[100, 100, 200, 200]]
                },
                "summary": "LPP Grade 2 detected with high confidence"
            }
        
        elif route == ProcessingRoute.MEDICAL_QUERY:
            return {
                "success": True,
                "route": route.value,
                "knowledge_result": {
                    "query_understood": True,
                    "response": "Protocol for Grade 2 LPP treatment...",
                    "confidence": 0.90
                },
                "summary": "Medical query processed successfully"
            }
        
        elif route == ProcessingRoute.HUMAN_REVIEW:
            return {
                "success": True,
                "route": route.value,
                "review_status": "queued",
                "queue_position": 3,
                "estimated_wait": "5 minutes",
                "summary": "Queued for human review"
            }
        
        elif route == ProcessingRoute.EMERGENCY:
            return {
                "success": True,
                "route": route.value,
                "escalation_status": "notified",
                "notified_parties": ["on_call_physician", "nurse_station"],
                "response_time": "immediate",
                "summary": "Emergency escalation completed"
            }
        
        else:  # INVALID
            return {
                "success": False,
                "route": route.value,
                "error": "Invalid input - no medical context",
                "summary": "Input rejected - non-medical content"
            }
    
    def _extract_patient_code(self, text: str) -> Optional[str]:
        """
        Extraer código de paciente del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Código de paciente si se encuentra y es válido
        """
        if not text:
            return None
        
        # Buscar patrón de código de paciente (ej: CD-2025-001)
        import re
        pattern = r'\b[A-Z]{2}-\d{4}-\d{3}\b'
        
        matches = re.findall(pattern, text.upper())
        
        for match in matches:
            if validate_patient_code_format(match):
                return match
        
        return None
    
    async def _tokenize_patient_data(self, hospital_mrn: str) -> TokenizedPatient:
        """
        Tokenizar datos de paciente usando el PHI Tokenization Service.
        Convierte Bruce Wayne → Batman para eliminar PHI del pipeline.
        
        Args:
            hospital_mrn: Medical Record Number del hospital (e.g., "MRN-2025-001-BW")
            
        Returns:
            TokenizedPatient con datos sin PHI
        """
        if not self.phi_client:
            raise Exception("PHI Tokenization client not initialized")
        
        # Determinar propósito basado en el contexto médico
        request_purpose = "Pressure injury detection and medical analysis"
        
        # Realizar tokenización (Bruce Wayne → Batman)
        tokenized_patient = await self.phi_client.tokenize_patient(
            hospital_mrn=hospital_mrn,
            request_purpose=request_purpose,
            urgency_level="routine"  # Se puede ajustar basado en flags de emergencia
        )
        
        return tokenized_patient
    
    def _route_to_session_type(self, route: ProcessingRoute) -> SessionType:
        """Mapear ruta de procesamiento a tipo de sesión."""
        mapping = {
            ProcessingRoute.CLINICAL_IMAGE: SessionType.CLINICAL_IMAGE,
            ProcessingRoute.MEDICAL_QUERY: SessionType.MEDICAL_QUERY,
            ProcessingRoute.HUMAN_REVIEW: SessionType.HUMAN_ESCALATION,
            ProcessingRoute.EMERGENCY: SessionType.EMERGENCY,
            ProcessingRoute.INVALID: SessionType.MEDICAL_QUERY  # Default
        }
        return mapping.get(route, SessionType.MEDICAL_QUERY)
    
    def register_route_processor(self, route: ProcessingRoute, processor: Any):
        """
        Registrar procesador para una ruta específica.
        
        Args:
            route: Ruta de procesamiento
            processor: Objeto procesador con método process()
        """
        self.route_processors[route] = processor
        logger.audit("route_processor_registered", {
            "route": route.value,
            "processor_class": processor.__class__.__name__
        })
    
    async def get_routing_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de routing."""
        return {
            "total_dispatched": self.routing_metrics["total_dispatched"],
            "by_route": self.routing_metrics["routes"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


    def _is_multimodal_medical_context(self, text_content: str) -> bool:
        """
        Detectar si el contexto requiere análisis multimodal (FASE 2).
        
        Busca indicadores que sugieran que tanto imagen como voz 
        son necesarios para una evaluación médica completa.
        
        Args:
            text_content: Contenido de texto del mensaje
            
        Returns:
            True si se requiere análisis multimodal
        """
        if not text_content:
            return False
        
        text_lower = text_content.lower()
        
        # Indicadores de análisis multimodal
        multimodal_indicators = [
            # Dolor y expresión vocal
            "dolor", "pain", "quejido", "gemido", "llanto",
            "crying", "moaning", "groaning", "sobbing",
            
            # Estrés y ansiedad vocal
            "ansiedad", "anxiety", "estrés", "stress", "nervioso",
            "nervous", "agitado", "agitated", "preocupado", "worried",
            
            # Contexto emocional
            "emocional", "emotional", "estado mental", "mental state",
            "psicológico", "psychological", "angustia", "anguish",
            
            # Evaluación comprehensiva
            "evaluación completa", "complete evaluation", "análisis total",
            "comprehensive analysis", "evaluación integral",
            
            # Indicadores de voz médica
            "voz", "voice", "habla", "speech", "expresión vocal",
            "vocal expression", "tono", "tone"
        ]
        
        # Verificar presencia de indicadores
        indicator_count = sum(
            1 for indicator in multimodal_indicators 
            if indicator in text_lower
        )
        
        # Contexto multimodal si hay 2+ indicadores o términos específicos
        if indicator_count >= 2:
            return True
        
        # Verificar frases específicas que requieren multimodal
        multimodal_phrases = [
            "evaluación completa del paciente",
            "análisis integral de",
            "estado emocional y físico",
            "dolor y estrés",
            "comprehensive patient assessment",
            "emotional and physical state",
            "pain and stress evaluation"
        ]
        
        return any(phrase in text_lower for phrase in multimodal_phrases)


class MedicalDispatcherFactory:
    """Factory para crear instancias de MedicalDispatcher."""
    
    @staticmethod
    async def create_dispatcher() -> MedicalDispatcher:
        """Crear y inicializar dispatcher."""
        dispatcher = MedicalDispatcher()
        await dispatcher.initialize()
        return dispatcher
    
    @staticmethod
    async def create_dispatcher_with_processors(processors: Dict[ProcessingRoute, Any]) -> MedicalDispatcher:
        """Crear dispatcher con procesadores pre-registrados."""
        dispatcher = MedicalDispatcher()
        await dispatcher.initialize()
        
        for route, processor in processors.items():
            dispatcher.register_route_processor(route, processor)
        
        return dispatcher