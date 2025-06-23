"""
Workflow Orchestration Agent - Agente ADK especializado en orquestación de flujos médicos
Implementa Agent Development Kit (ADK) para gestión de workflows y coordinación médica.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

from .base_agent import BaseAgent, AgentCapability, AgentMessage, AgentResponse
from ..core.medical_dispatcher import MedicalDispatcher, ProcessingRoute, TriageDecision
from ..core.input_packager import StandardizedInput
from ..core.session_manager import SessionManager, SessionState, SessionType
from ..utils.secure_logger import SecureLogger

logger = SecureLogger("workflow_orchestration_agent")


class WorkflowType(Enum):
    """Tipos de workflow médico."""
    CLINICAL_ASSESSMENT = "clinical_assessment"
    EMERGENCY_RESPONSE = "emergency_response"
    ROUTINE_REVIEW = "routine_review"
    PROTOCOL_EXECUTION = "protocol_execution"
    MULTI_SPECIALIST_CONSULTATION = "multi_specialist_consultation"
    AUDIT_TRAIL = "audit_trail"


class WorkflowStage(Enum):
    """Etapas de workflow."""
    INITIATED = "initiated"
    TRIAGING = "triaging"
    PROCESSING = "processing"
    REVIEW = "review"
    ESCALATION = "escalation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowPriority(Enum):
    """Prioridades de workflow."""
    CRITICAL = "critical"      # Emergencias - procesamiento inmediato
    HIGH = "high"             # Urgente - < 30 min
    MEDIUM = "medium"         # Rutina - < 2 horas
    LOW = "low"              # No urgente - < 24 horas


@dataclass
class WorkflowRequest:
    """Solicitud de orquestación de workflow."""
    workflow_type: WorkflowType
    priority: WorkflowPriority
    input_data: Dict[str, Any]
    required_stages: List[WorkflowStage]
    timeout_minutes: int = 30
    requires_human_review: bool = False
    escalation_rules: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowExecution:
    """Ejecución de workflow en progreso."""
    workflow_id: str
    workflow_type: WorkflowType
    current_stage: WorkflowStage
    completed_stages: List[WorkflowStage]
    pending_stages: List[WorkflowStage]
    start_time: datetime
    last_update: datetime
    timeout_at: datetime
    results: Dict[str, Any]
    errors: List[str]
    requires_intervention: bool


@dataclass
class WorkflowResult:
    """Resultado de ejecución de workflow."""
    success: bool
    workflow_id: str
    final_stage: WorkflowStage
    execution_time: float
    stage_results: Dict[str, Any]
    medical_outcome: Dict[str, Any]
    escalations_triggered: List[str]
    next_actions: List[str]


class WorkflowOrchestrationAgent(BaseAgent):
    """
    Agent especializado en orquestación de workflows médicos.
    
    Responsabilidades:
    - Coordinar flujos de trabajo médicos complejos
    - Gestionar transiciones entre etapas
    - Implementar timeouts médicos críticos
    - Coordinar múltiples agentes especializados
    - Mantener trazabilidad completa del workflow
    """
    
    def __init__(self, 
                 agent_id: str = "workflow_orchestration_agent",
                 medical_dispatcher: Optional[MedicalDispatcher] = None,
                 session_manager: Optional[SessionManager] = None):
        """
        Inicializar WorkflowOrchestrationAgent.
        
        Args:
            agent_id: Identificador único del agente
            medical_dispatcher: Dispatcher médico
            session_manager: Gestor de sesiones
        """
        super().__init__(
            agent_id=agent_id,
            name="Workflow Orchestration Agent",
            description="Agente especializado en orquestación de workflows médicos",
            capabilities=[
                AgentCapability.WORKFLOW_ORCHESTRATION,
                AgentCapability.MEDICAL_COORDINATION,
                AgentCapability.TIMEOUT_MANAGEMENT,
                AgentCapability.ESCALATION_MANAGEMENT
            ],
            version="1.0.0"
        )
        
        # Componentes core
        self.medical_dispatcher = medical_dispatcher or MedicalDispatcher()
        self.session_manager = session_manager or SessionManager()
        
        # Configuración de workflows
        self.workflow_config = {
            "max_concurrent_workflows": 50,
            "default_timeout_minutes": 30,
            "escalation_timeout_minutes": 60,
            "retry_attempts": 3,
            "stage_timeout_seconds": 300  # 5 minutos por etapa
        }
        
        # Workflows activos
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        
        # Métricas de workflow
        self.workflow_metrics = {
            "total_executed": 0,
            "successful": 0,
            "failed": 0,
            "escalated": 0,
            "average_execution_time": 0.0,
            "by_type": {wt.value: 0 for wt in WorkflowType}
        }
        
        logger.audit("workflow_orchestration_agent_initialized", {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "max_concurrent": self.workflow_config["max_concurrent_workflows"],
            "default_timeout": self.workflow_config["default_timeout_minutes"]
        })
    
    async def initialize(self) -> bool:
        """
        Inicializar el agente y sus dependencias.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Inicializar componentes
            await self.medical_dispatcher.initialize()
            await self.session_manager.initialize()
            
            # Marcar como inicializado
            self.is_initialized = True
            
            logger.audit("workflow_orchestration_agent_ready", {
                "agent_id": self.agent_id,
                "initialization_successful": True,
                "medical_dispatcher_ready": True,
                "session_manager_ready": True
            })
            
            return True
            
        except Exception as e:
            logger.error("workflow_orchestration_agent_initialization_failed", {
                "agent_id": self.agent_id,
                "error": str(e)
            })
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Procesar mensaje y orquestar workflow médico.
        
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
                    "Invalid message format for workflow orchestration"
                )
            
            # Extraer solicitud de workflow
            workflow_request = self._extract_workflow_request(message)
            
            # Ejecutar workflow
            workflow_result = await self._execute_workflow(workflow_request, message.session_id)
            
            # Generar respuesta
            response = await self._generate_workflow_response(
                message,
                workflow_request, 
                workflow_result,
                start_time
            )
            
            # Actualizar métricas
            self._update_metrics(workflow_request, workflow_result)
            
            # Registrar ejecución de workflow
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.audit("workflow_executed", {
                "agent_id": self.agent_id,
                "session_id": message.session_id,
                "workflow_type": workflow_request.workflow_type.value,
                "workflow_id": workflow_result.workflow_id,
                "success": workflow_result.success,
                "execution_time": workflow_result.execution_time,
                "final_stage": workflow_result.final_stage.value,
                "processing_time": processing_time
            })
            
            return response
            
        except Exception as e:
            logger.error("workflow_orchestration_failed", {
                "agent_id": self.agent_id,
                "session_id": message.session_id,
                "error": str(e)
            })
            
            return self._create_error_response(message, str(e))
    
    async def _execute_workflow(self, 
                               request: WorkflowRequest, 
                               session_id: str) -> WorkflowResult:
        """
        Ejecutar workflow médico completo.
        
        Args:
            request: Solicitud de workflow
            session_id: ID de sesión
            
        Returns:
            WorkflowResult: Resultado de la ejecución
        """
        # Generar ID único de workflow
        workflow_id = f"{request.workflow_type.value}_{session_id}_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Crear ejecución de workflow
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                workflow_type=request.workflow_type,
                current_stage=WorkflowStage.INITIATED,
                completed_stages=[],
                pending_stages=request.required_stages.copy(),
                start_time=start_time,
                last_update=start_time,
                timeout_at=start_time + timedelta(minutes=request.timeout_minutes),
                results={},
                errors=[],
                requires_intervention=False
            )
            
            # Registrar workflow activo
            self.active_workflows[workflow_id] = execution
            
            # Ejecutar etapas del workflow
            for stage in request.required_stages:
                stage_result = await self._execute_workflow_stage(
                    execution, 
                    stage, 
                    request
                )
                
                # Actualizar ejecución
                execution.current_stage = stage
                execution.completed_stages.append(stage)
                execution.pending_stages.remove(stage)
                execution.last_update = datetime.now(timezone.utc)
                execution.results[stage.value] = stage_result
                
                # Verificar si la etapa falló
                if not stage_result.get("success", False):
                    execution.errors.append(f"Stage {stage.value} failed: {stage_result.get('error', 'Unknown error')}")
                    
                    # Determinar si continuar o fallar
                    if stage_result.get("critical_failure", False):
                        execution.current_stage = WorkflowStage.FAILED
                        break
                    elif stage_result.get("requires_escalation", False):
                        execution.requires_intervention = True
                        execution.current_stage = WorkflowStage.ESCALATION
                        break
                
                # Verificar timeout
                if datetime.now(timezone.utc) > execution.timeout_at:
                    execution.errors.append("Workflow timeout exceeded")
                    execution.current_stage = WorkflowStage.FAILED
                    break
            
            # Determinar estado final
            if execution.current_stage not in [WorkflowStage.FAILED, WorkflowStage.ESCALATION]:
                execution.current_stage = WorkflowStage.COMPLETED
            
            # Calcular tiempo de ejecución
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Limpiar workflow activo
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Generar resultado final
            return WorkflowResult(
                success=execution.current_stage == WorkflowStage.COMPLETED,
                workflow_id=workflow_id,
                final_stage=execution.current_stage,
                execution_time=execution_time,
                stage_results=execution.results,
                medical_outcome=self._extract_medical_outcome(execution),
                escalations_triggered=self._extract_escalations(execution),
                next_actions=self._determine_next_actions(execution)
            )
            
        except Exception as e:
            # Limpiar workflow en caso de error
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            logger.error("workflow_execution_failed", {
                "workflow_id": workflow_id,
                "workflow_type": request.workflow_type.value,
                "error": str(e)
            })
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return WorkflowResult(
                success=False,
                workflow_id=workflow_id,
                final_stage=WorkflowStage.FAILED,
                execution_time=execution_time,
                stage_results={},
                medical_outcome={"error": str(e)},
                escalations_triggered=["technical_failure"],
                next_actions=["Reintentar workflow", "Escalación técnica"]
            )
    
    async def _execute_workflow_stage(self, 
                                    execution: WorkflowExecution,
                                    stage: WorkflowStage,
                                    request: WorkflowRequest) -> Dict[str, Any]:
        """
        Ejecutar una etapa específica del workflow.
        
        Args:
            execution: Ejecución de workflow
            stage: Etapa a ejecutar
            request: Solicitud original
            
        Returns:
            Dict con resultado de la etapa
        """
        try:
            logger.info("workflow_stage_start", {
                "workflow_id": execution.workflow_id,
                "stage": stage.value,
                "workflow_type": execution.workflow_type.value
            })
            
            # Timeout para la etapa
            stage_timeout = self.workflow_config["stage_timeout_seconds"]
            
            # Ejecutar según el tipo de etapa
            if stage == WorkflowStage.TRIAGING:
                result = await self._execute_triage_stage(execution, request)
            elif stage == WorkflowStage.PROCESSING:
                result = await self._execute_processing_stage(execution, request)
            elif stage == WorkflowStage.REVIEW:
                result = await self._execute_review_stage(execution, request)
            elif stage == WorkflowStage.ESCALATION:
                result = await self._execute_escalation_stage(execution, request)
            else:
                # Etapa genérica
                result = await self._execute_generic_stage(execution, stage, request)
            
            logger.audit("workflow_stage_completed", {
                "workflow_id": execution.workflow_id,
                "stage": stage.value,
                "success": result.get("success", False),
                "duration": result.get("duration", 0)
            })
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("workflow_stage_timeout", {
                "workflow_id": execution.workflow_id,
                "stage": stage.value,
                "timeout_seconds": stage_timeout
            })
            
            return {
                "success": False,
                "error": f"Stage {stage.value} timeout after {stage_timeout} seconds",
                "requires_escalation": True,
                "duration": stage_timeout
            }
            
        except Exception as e:
            logger.error("workflow_stage_failed", {
                "workflow_id": execution.workflow_id,
                "stage": stage.value,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e),
                "critical_failure": True,
                "duration": 0
            }
    
    async def _execute_triage_stage(self, 
                                  execution: WorkflowExecution,
                                  request: WorkflowRequest) -> Dict[str, Any]:
        """Ejecutar etapa de triage médico."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Crear StandardizedInput para el dispatcher
            standardized_input = StandardizedInput(
                session_id=execution.workflow_id,
                input_type=request.input_data.get("input_type", "text"),
                content=request.input_data.get("content", {}),
                raw_content=request.input_data.get("raw_content", {}),
                metadata=request.input_data.get("metadata", {}),
                security_context=request.input_data.get("security_context", {
                    "encryption_enabled": True,
                    "phi_detected": False
                })
            )
            
            # Ejecutar dispatch
            dispatch_result = await self.medical_dispatcher.dispatch(standardized_input)
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "success": dispatch_result.get("success", False),
                "triage_decision": dispatch_result.get("route"),
                "confidence": dispatch_result.get("triage_confidence"),
                "dispatch_result": dispatch_result,
                "duration": duration
            }
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "requires_escalation": True,
                "duration": duration
            }
    
    async def _execute_processing_stage(self, 
                                      execution: WorkflowExecution,
                                      request: WorkflowRequest) -> Dict[str, Any]:
        """Ejecutar etapa de procesamiento médico."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Obtener resultado del triage
            triage_result = execution.results.get("triaging", {})
            route = triage_result.get("triage_decision")
            
            # Simular procesamiento específico por ruta
            if route == "clinical_image_processing":
                result = await self._process_clinical_image(request.input_data)
            elif route == "medical_knowledge_system":
                result = await self._process_medical_query(request.input_data)
            elif route == "human_review_queue":
                result = await self._queue_human_review(request.input_data)
            elif route == "emergency_escalation":
                result = await self._escalate_emergency(request.input_data)
            else:
                result = {"success": False, "error": "Unknown processing route"}
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result["duration"] = duration
            
            return result
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "critical_failure": True,
                "duration": duration
            }
    
    async def _execute_review_stage(self, 
                                  execution: WorkflowExecution,
                                  request: WorkflowRequest) -> Dict[str, Any]:
        """Ejecutar etapa de revisión médica."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Obtener resultado de procesamiento
            processing_result = execution.results.get("processing", {})
            
            # Determinar si requiere revisión humana
            requires_human_review = (
                request.requires_human_review or
                processing_result.get("confidence", 1.0) < 0.7 or
                processing_result.get("error") is not None
            )
            
            if requires_human_review:
                # Simular cola de revisión humana
                review_result = {
                    "success": True,
                    "review_status": "queued",
                    "queue_position": 2,
                    "estimated_wait_minutes": 15,
                    "reviewer_assigned": "medical_team"
                }
            else:
                # Auto-aprobación
                review_result = {
                    "success": True,
                    "review_status": "auto_approved",
                    "confidence": processing_result.get("confidence", 0.95),
                    "reviewer": "automated_system"
                }
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            review_result["duration"] = duration
            
            return review_result
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "requires_escalation": True,
                "duration": duration
            }
    
    async def _execute_escalation_stage(self, 
                                      execution: WorkflowExecution,
                                      request: WorkflowRequest) -> Dict[str, Any]:
        """Ejecutar etapa de escalamiento."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Determinar tipo de escalamiento basado en errores
            escalation_type = "technical"
            if any("medical" in error.lower() for error in execution.errors):
                escalation_type = "medical"
            elif any("emergency" in error.lower() for error in execution.errors):
                escalation_type = "emergency"
            
            # Simular escalamiento
            escalation_result = {
                "success": True,
                "escalation_type": escalation_type,
                "notified_parties": [
                    "on_call_physician" if escalation_type == "medical" else "technical_support",
                    "supervisor"
                ],
                "escalation_id": f"ESC_{execution.workflow_id}_{int(datetime.now().timestamp())}",
                "priority": "high" if escalation_type == "emergency" else "medium"
            }
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            escalation_result["duration"] = duration
            
            return escalation_result
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "critical_failure": True,
                "duration": duration
            }
    
    async def _execute_generic_stage(self, 
                                   execution: WorkflowExecution,
                                   stage: WorkflowStage,
                                   request: WorkflowRequest) -> Dict[str, Any]:
        """Ejecutar etapa genérica."""
        start_time = datetime.now(timezone.utc)
        
        # Simular procesamiento genérico
        await asyncio.sleep(1)  # Simular trabajo
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            "success": True,
            "stage": stage.value,
            "status": "completed",
            "duration": duration
        }
    
    # Métodos auxiliares para procesamiento específico
    async def _process_clinical_image(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simular procesamiento de imagen clínica."""
        await asyncio.sleep(2)  # Simular análisis de imagen
        return {
            "success": True,
            "lpp_detected": True,
            "lpp_grade": 2,
            "confidence": 0.87,
            "anatomical_location": "sacrum",
            "recommendations": ["Cambio postural cada 2h", "Apósito especializado"]
        }
    
    async def _process_medical_query(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simular procesamiento de consulta médica."""
        await asyncio.sleep(1)  # Simular búsqueda en base de conocimiento
        return {
            "success": True,
            "query_understood": True,
            "response": "Protocolo de tratamiento LPP encontrado",
            "confidence": 0.92,
            "references": ["NPUAP Guidelines 2019"]
        }
    
    async def _queue_human_review(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simular cola de revisión humana."""
        return {
            "success": True,
            "queue_status": "added",
            "queue_position": 3,
            "estimated_wait": "10 minutes"
        }
    
    async def _escalate_emergency(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simular escalamiento de emergencia."""
        return {
            "success": True,
            "escalation_status": "emergency_notified",
            "response_time": "immediate",
            "notified_parties": ["emergency_team", "on_call_physician"]
        }
    
    def _extract_medical_outcome(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Extraer resultado médico del workflow."""
        medical_outcome = {}
        
        # Extraer de resultados de procesamiento
        processing_result = execution.results.get("processing", {})
        if processing_result.get("lpp_detected"):
            medical_outcome["diagnosis"] = {
                "condition": "LPP",
                "grade": processing_result.get("lpp_grade"),
                "confidence": processing_result.get("confidence"),
                "location": processing_result.get("anatomical_location")
            }
            medical_outcome["recommendations"] = processing_result.get("recommendations", [])
        
        # Extraer de revisión
        review_result = execution.results.get("review", {})
        if review_result.get("review_status"):
            medical_outcome["review_status"] = review_result["review_status"]
        
        return medical_outcome
    
    def _extract_escalations(self, execution: WorkflowExecution) -> List[str]:
        """Extraer escalamientos del workflow."""
        escalations = []
        
        # Buscar escalamientos en todas las etapas
        for stage_result in execution.results.values():
            if stage_result.get("escalation_type"):
                escalations.append(stage_result["escalation_type"])
            if stage_result.get("requires_escalation"):
                escalations.append("stage_escalation")
        
        return escalations
    
    def _determine_next_actions(self, execution: WorkflowExecution) -> List[str]:
        """Determinar próximas acciones basadas en el resultado."""
        next_actions = []
        
        if execution.current_stage == WorkflowStage.COMPLETED:
            # Workflow completado exitosamente
            medical_outcome = self._extract_medical_outcome(execution)
            if medical_outcome.get("diagnosis"):
                next_actions.extend([
                    "Implementar recomendaciones médicas",
                    "Programar seguimiento",
                    "Documentar en historia clínica"
                ])
        elif execution.current_stage == WorkflowStage.FAILED:
            next_actions.extend([
                "Revisar errores del workflow",
                "Reintentar procesamiento",
                "Escalación manual si es necesario"
            ])
        elif execution.current_stage == WorkflowStage.ESCALATION:
            next_actions.extend([
                "Esperar respuesta de escalamiento",
                "Monitorear estado de revisión",
                "Preparar información adicional"
            ])
        
        return next_actions
    
    def _extract_workflow_request(self, message: AgentMessage) -> WorkflowRequest:
        """Extraer solicitud de workflow del mensaje."""
        content = message.content
        
        # Determinar tipo de workflow
        workflow_type = self._classify_workflow_type(content.get("text", ""))
        
        # Determinar prioridad
        priority = self._determine_priority(content)
        
        # Extraer datos de entrada
        input_data = content.get("input_data", content)
        
        # Determinar etapas requeridas
        required_stages = self._determine_required_stages(workflow_type, content)
        
        # Extraer timeout
        timeout_minutes = content.get("timeout_minutes", self.workflow_config["default_timeout_minutes"])
        
        # Determinar si requiere revisión humana
        requires_human_review = content.get("requires_human_review", False)
        
        return WorkflowRequest(
            workflow_type=workflow_type,
            priority=priority,
            input_data=input_data,
            required_stages=required_stages,
            timeout_minutes=timeout_minutes,
            requires_human_review=requires_human_review,
            escalation_rules=content.get("escalation_rules"),
            metadata=content.get("metadata", {})
        )
    
    def _classify_workflow_type(self, text: str) -> WorkflowType:
        """Clasificar tipo de workflow basado en el texto."""
        text_lower = text.lower()
        
        # Palabras clave por tipo
        type_keywords = {
            WorkflowType.CLINICAL_ASSESSMENT: [
                "imagen", "image", "lpp", "lesión", "injury", "diagnóstico", "diagnosis"
            ],
            WorkflowType.EMERGENCY_RESPONSE: [
                "emergencia", "emergency", "crítico", "critical", "urgente", "urgent"
            ],
            WorkflowType.ROUTINE_REVIEW: [
                "revisión", "review", "rutina", "routine", "seguimiento", "follow"
            ],
            WorkflowType.PROTOCOL_EXECUTION: [
                "protocolo", "protocol", "procedimiento", "procedure", "guía", "guideline"
            ],
            WorkflowType.MULTI_SPECIALIST_CONSULTATION: [
                "consulta", "consultation", "especialista", "specialist", "múltiple", "multi"
            ],
            WorkflowType.AUDIT_TRAIL: [
                "auditoría", "audit", "trazabilidad", "trail", "compliance", "cumplimiento"
            ]
        }
        
        # Buscar coincidencias
        for workflow_type, keywords in type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return workflow_type
        
        # Default
        return WorkflowType.CLINICAL_ASSESSMENT
    
    def _determine_priority(self, content: Dict[str, Any]) -> WorkflowPriority:
        """Determinar prioridad del workflow."""
        priority_str = content.get("priority", "medium").lower()
        
        priority_map = {
            "critical": WorkflowPriority.CRITICAL,
            "high": WorkflowPriority.HIGH,
            "medium": WorkflowPriority.MEDIUM,
            "low": WorkflowPriority.LOW
        }
        
        return priority_map.get(priority_str, WorkflowPriority.MEDIUM)
    
    def _determine_required_stages(self, 
                                 workflow_type: WorkflowType, 
                                 content: Dict[str, Any]) -> List[WorkflowStage]:
        """Determinar etapas requeridas para el workflow."""
        # Etapas por defecto por tipo de workflow
        default_stages = {
            WorkflowType.CLINICAL_ASSESSMENT: [
                WorkflowStage.TRIAGING,
                WorkflowStage.PROCESSING,
                WorkflowStage.REVIEW
            ],
            WorkflowType.EMERGENCY_RESPONSE: [
                WorkflowStage.TRIAGING,
                WorkflowStage.PROCESSING,
                WorkflowStage.ESCALATION
            ],
            WorkflowType.ROUTINE_REVIEW: [
                WorkflowStage.PROCESSING,
                WorkflowStage.REVIEW
            ],
            WorkflowType.PROTOCOL_EXECUTION: [
                WorkflowStage.TRIAGING,
                WorkflowStage.PROCESSING
            ],
            WorkflowType.MULTI_SPECIALIST_CONSULTATION: [
                WorkflowStage.TRIAGING,
                WorkflowStage.PROCESSING,
                WorkflowStage.REVIEW,
                WorkflowStage.ESCALATION
            ],
            WorkflowType.AUDIT_TRAIL: [
                WorkflowStage.PROCESSING
            ]
        }
        
        # Permitir override desde el contenido
        if "required_stages" in content:
            stage_names = content["required_stages"]
            return [WorkflowStage(stage) for stage in stage_names if stage in [s.value for s in WorkflowStage]]
        
        return default_stages.get(workflow_type, [WorkflowStage.PROCESSING])
    
    def _update_metrics(self, request: WorkflowRequest, result: WorkflowResult):
        """Actualizar métricas de workflow."""
        self.workflow_metrics["total_executed"] += 1
        self.workflow_metrics["by_type"][request.workflow_type.value] += 1
        
        if result.success:
            self.workflow_metrics["successful"] += 1
        else:
            self.workflow_metrics["failed"] += 1
        
        if result.escalations_triggered:
            self.workflow_metrics["escalated"] += 1
        
        # Actualizar tiempo promedio
        total = self.workflow_metrics["total_executed"]
        current_avg = self.workflow_metrics["average_execution_time"]
        new_avg = ((current_avg * (total - 1)) + result.execution_time) / total
        self.workflow_metrics["average_execution_time"] = new_avg
    
    async def _generate_workflow_response(self, 
                                        message: AgentMessage,
                                        request: WorkflowRequest,
                                        result: WorkflowResult,
                                        start_time: datetime) -> AgentResponse:
        """Generar respuesta de workflow."""
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        if result.success:
            response_content = {
                "workflow_result": asdict(result),
                "status": "completed",
                "medical_outcome": result.medical_outcome,
                "next_actions": result.next_actions
            }
            
            success = True
            response_text = f"Workflow {request.workflow_type.value} completado exitosamente"
            
        else:
            response_content = {
                "workflow_result": asdict(result),
                "status": "failed",
                "final_stage": result.final_stage.value,
                "escalations": result.escalations_triggered,
                "next_actions": result.next_actions
            }
            
            success = False
            response_text = f"Workflow {request.workflow_type.value} falló en etapa {result.final_stage.value}"
        
        return AgentResponse(
            session_id=message.session_id,
            agent_id=self.agent_id,
            success=success,
            content=response_content,
            metadata={
                "processing_time": processing_time,
                "workflow_type": request.workflow_type.value,
                "workflow_id": result.workflow_id,
                "execution_time": result.execution_time,
                "final_stage": result.final_stage.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            message=response_text,
            next_actions=result.next_actions,
            requires_human_review=not result.success or bool(result.escalations_triggered)
        )
    
    def _validate_message(self, message: AgentMessage) -> bool:
        """Validar formato del mensaje."""
        if not message.content:
            return False
        
        # Debe tener datos de entrada
        return bool(message.content.get("text") or message.content.get("input_data"))
    
    def _create_error_response(self, message: AgentMessage, error: str) -> AgentResponse:
        """Crear respuesta de error."""
        return AgentResponse(
            session_id=message.session_id,
            agent_id=self.agent_id,
            success=False,
            content={
                "error": error,
                "status": "error"
            },
            metadata={
                "error_occurred": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            message=f"Error en orquestación de workflow: {error}",
            next_actions=["Verificar datos de entrada", "Reintentar workflow"],
            requires_human_review=True
        )
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de workflows."""
        return {
            **self.workflow_metrics,
            "active_workflows": len(self.active_workflows),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Obtener workflows activos."""
        return {
            wf_id: {
                "workflow_type": execution.workflow_type.value,
                "current_stage": execution.current_stage.value,
                "progress": len(execution.completed_stages) / (len(execution.completed_stages) + len(execution.pending_stages)) * 100,
                "start_time": execution.start_time.isoformat(),
                "timeout_at": execution.timeout_at.isoformat()
            }
            for wf_id, execution in self.active_workflows.items()
        }
    
    async def shutdown(self):
        """Cerrar agente y liberar recursos."""
        try:
            logger.audit("workflow_orchestration_agent_shutdown", {
                "agent_id": self.agent_id,
                "active_workflows": len(self.active_workflows),
                "total_executed": self.workflow_metrics["total_executed"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Limpiar workflows activos
            self.active_workflows.clear()
            
            self.is_initialized = False
            
        except Exception as e:
            logger.error("workflow_orchestration_agent_shutdown_error", {
                "agent_id": self.agent_id,
                "error": str(e)
            })


# Factory para crear WorkflowOrchestrationAgent
class WorkflowOrchestrationAgentFactory:
    """Factory para crear instancias de WorkflowOrchestrationAgent."""
    
    @staticmethod
    async def create_agent(config: Optional[Dict[str, Any]] = None) -> WorkflowOrchestrationAgent:
        """
        Crear instancia de WorkflowOrchestrationAgent.
        
        Args:
            config: Configuración opcional
            
        Returns:
            WorkflowOrchestrationAgent: Instancia configurada
        """
        config = config or {}
        
        # Crear componentes si se especifica
        medical_dispatcher = None
        session_manager = None
        
        if config.get("use_medical_dispatcher", True):
            medical_dispatcher = MedicalDispatcher()
            
        if config.get("use_session_manager", True):
            session_manager = SessionManager()
        
        # Crear agente
        agent = WorkflowOrchestrationAgent(
            agent_id=config.get("agent_id", "workflow_orchestration_agent"),
            medical_dispatcher=medical_dispatcher,
            session_manager=session_manager
        )
        
        # Inicializar
        await agent.initialize()
        
        return agent