"""
Matriz de Control de Acceso para Arquitectura 3 Capas Segura Vigia.

Este módulo implementa los controles de acceso y permisos para cada capa
del sistema, garantizando el aislamiento de seguridad y cumplimiento normativo.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass

from vigia_detect.utils.audit_service import AuditService, AuditEvent, AuditLevel


class SystemLayer(Enum):
    """Capas del sistema con diferentes niveles de acceso."""
    LAYER_1_INPUT = "layer_1_input"           # Capa de entrada aislada
    LAYER_2_ORCHESTRATION = "layer_2_orchestration"  # Capa de orquestación médica
    LAYER_3_SPECIALIZED = "layer_3_specialized"      # Capa de sistemas especializados
    CROSS_CUTTING = "cross_cutting"           # Servicios transversales


class AccessLevel(Enum):
    """Niveles de acceso del sistema."""
    NO_ACCESS = 0        # Sin acceso
    READ_ONLY = 1        # Solo lectura
    READ_WRITE = 2       # Lectura y escritura
    ADMIN = 3            # Acceso administrativo
    SYSTEM = 4           # Acceso de sistema (máximo)


class ResourceType(Enum):
    """Tipos de recursos en el sistema."""
    MEDICAL_DATA = "medical_data"              # Datos médicos sensibles
    PATIENT_INFO = "patient_info"              # Información de pacientes
    CLINICAL_IMAGES = "clinical_images"        # Imágenes clínicas
    DIAGNOSTIC_RESULTS = "diagnostic_results"  # Resultados de diagnóstico
    AUDIT_LOGS = "audit_logs"                 # Logs de auditoría
    SYSTEM_CONFIG = "system_config"           # Configuración del sistema
    SESSION_DATA = "session_data"             # Datos de sesión
    ENCRYPTION_KEYS = "encryption_keys"       # Claves de encriptación
    PROTOCOL_DATA = "protocol_data"           # Protocolos médicos
    NOTIFICATION_DATA = "notification_data"   # Datos de notificaciones


class ComponentRole(Enum):
    """Roles de componentes del sistema."""
    # Capa 1 - Input Layer (Zero Medical Knowledge)
    WHATSAPP_BOT = "whatsapp_bot"
    INPUT_PACKAGER = "input_packager" 
    INPUT_QUEUE = "input_queue"
    
    # Capa 2 - Orchestration Layer
    MEDICAL_DISPATCHER = "medical_dispatcher"
    TRIAGE_ENGINE = "triage_engine"
    SESSION_MANAGER = "session_manager"
    
    # Capa 3 - Specialized Systems
    CLINICAL_PROCESSOR = "clinical_processor"
    MEDICAL_KNOWLEDGE = "medical_knowledge"
    HUMAN_REVIEW_QUEUE = "human_review_queue"
    
    # Cross-cutting
    AUDIT_SERVICE = "audit_service"
    SLACK_ORCHESTRATOR = "slack_orchestrator"
    
    # Usuarios externos
    MEDICAL_STAFF = "medical_staff"
    ADMIN_USER = "admin_user"
    SYSTEM_MONITOR = "system_monitor"


@dataclass
class AccessRule:
    """Regla de acceso específica."""
    component: ComponentRole
    resource: ResourceType
    access_level: AccessLevel
    conditions: Optional[Dict[str, Any]] = None
    time_restrictions: Optional[Dict[str, Any]] = None
    audit_required: bool = True


@dataclass
class AccessRequest:
    """Solicitud de acceso a recurso."""
    component: ComponentRole
    resource: ResourceType
    operation: str  # read, write, delete, etc.
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AccessDecision:
    """Decisión de acceso."""
    granted: bool
    reason: str
    access_level: AccessLevel
    conditions: Optional[Dict[str, Any]] = None
    audit_event: Optional[AuditEvent] = None


class AccessControlMatrix:
    """
    Matriz de Control de Acceso para el Sistema Vigia.
    
    Implementa los principios de:
    - Principio de menor privilegio
    - Aislamiento por capas
    - Auditoría completa
    - Controles temporales
    - Validación de contexto médico
    """
    
    def __init__(self):
        self.audit_service = AuditService()
        self.logger = logging.getLogger(__name__)
        
        # Matriz de permisos por componente y recurso
        self.access_matrix = self._initialize_access_matrix()
        
        # Sesiones activas con permisos temporales
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Cache de decisiones de acceso (para performance)
        self.decision_cache: Dict[str, AccessDecision] = {}
        self.cache_ttl = timedelta(minutes=5)
    
    def _initialize_access_matrix(self) -> Dict[ComponentRole, Dict[ResourceType, AccessRule]]:
        """Inicializa la matriz de control de acceso por capa."""
        matrix = {}
        
        # === CAPA 1: INPUT LAYER (Zero Medical Knowledge) ===
        
        # WhatsApp Bot - NO acceso a datos médicos
        matrix[ComponentRole.WHATSAPP_BOT] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.WHATSAPP_BOT, ResourceType.MEDICAL_DATA, 
                AccessLevel.NO_ACCESS, audit_required=True
            ),
            ResourceType.PATIENT_INFO: AccessRule(
                ComponentRole.WHATSAPP_BOT, ResourceType.PATIENT_INFO,
                AccessLevel.NO_ACCESS, audit_required=True
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.WHATSAPP_BOT, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_ONLY,  # Solo para transferir, no procesar
                conditions={"encryption_required": True, "no_medical_analysis": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.WHATSAPP_BOT, ResourceType.SESSION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"session_scope_only": True}
            ),
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.WHATSAPP_BOT, ResourceType.AUDIT_LOGS,
                AccessLevel.READ_ONLY, conditions={"own_events_only": True}
            )
        }
        
        # Input Packager - Estandarización sin conocimiento médico
        matrix[ComponentRole.INPUT_PACKAGER] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.INPUT_PACKAGER, ResourceType.MEDICAL_DATA,
                AccessLevel.NO_ACCESS, audit_required=True
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.INPUT_PACKAGER, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_ONLY,
                conditions={"format_standardization_only": True, "no_content_analysis": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.INPUT_PACKAGER, ResourceType.SESSION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"metadata_only": True}
            ),
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.INPUT_PACKAGER, ResourceType.AUDIT_LOGS,
                AccessLevel.READ_WRITE, conditions={"create_events_only": True}
            )
        }
        
        # Input Queue - Almacenamiento encriptado
        matrix[ComponentRole.INPUT_QUEUE] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.INPUT_QUEUE, ResourceType.MEDICAL_DATA,
                AccessLevel.NO_ACCESS, audit_required=True
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.INPUT_QUEUE, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_WRITE,
                conditions={"encrypted_storage_only": True, "no_processing": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.INPUT_QUEUE, ResourceType.SESSION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"queue_management_only": True}
            ),
            ResourceType.ENCRYPTION_KEYS: AccessRule(
                ComponentRole.INPUT_QUEUE, ResourceType.ENCRYPTION_KEYS,
                AccessLevel.READ_ONLY,
                conditions={"queue_encryption_only": True}
            )
        }
        
        # === CAPA 2: ORCHESTRATION LAYER ===
        
        # Medical Dispatcher - Enrutamiento con conocimiento médico limitado
        matrix[ComponentRole.MEDICAL_DISPATCHER] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.MEDICAL_DISPATCHER, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"routing_decisions_only": True, "no_detailed_analysis": True}
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.MEDICAL_DISPATCHER, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_ONLY,
                conditions={"routing_metadata_only": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.MEDICAL_DISPATCHER, ResourceType.SESSION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"orchestration_context": True}
            ),
            ResourceType.PROTOCOL_DATA: AccessRule(
                ComponentRole.MEDICAL_DISPATCHER, ResourceType.PROTOCOL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"routing_protocols_only": True}
            )
        }
        
        # Triage Engine - Análisis de urgencia médica
        matrix[ComponentRole.TRIAGE_ENGINE] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.TRIAGE_ENGINE, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"urgency_assessment_only": True, "limited_scope": True}
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.TRIAGE_ENGINE, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_ONLY,
                conditions={"triage_analysis_only": True}
            ),
            ResourceType.PROTOCOL_DATA: AccessRule(
                ComponentRole.TRIAGE_ENGINE, ResourceType.PROTOCOL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"triage_rules_only": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.TRIAGE_ENGINE, ResourceType.SESSION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"triage_context": True}
            )
        }
        
        # Session Manager - Gestión de sesiones y aislamiento temporal
        matrix[ComponentRole.SESSION_MANAGER] = {
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.SESSION_MANAGER, ResourceType.SESSION_DATA,
                AccessLevel.ADMIN,
                conditions={"session_lifecycle_management": True}
            ),
            ResourceType.ENCRYPTION_KEYS: AccessRule(
                ComponentRole.SESSION_MANAGER, ResourceType.ENCRYPTION_KEYS,
                AccessLevel.READ_ONLY,
                conditions={"session_encryption_only": True}
            ),
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.SESSION_MANAGER, ResourceType.MEDICAL_DATA,
                AccessLevel.NO_ACCESS, audit_required=True
            ),
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.SESSION_MANAGER, ResourceType.AUDIT_LOGS,
                AccessLevel.READ_WRITE,
                conditions={"session_events_only": True}
            )
        }
        
        # === CAPA 3: SPECIALIZED SYSTEMS ===
        
        # Clinical Processor - Análisis clínico completo
        matrix[ComponentRole.CLINICAL_PROCESSOR] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.CLINICAL_PROCESSOR, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_WRITE,
                conditions={"clinical_analysis_context": True, "session_scoped": True}
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.CLINICAL_PROCESSOR, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_WRITE,
                conditions={"lpp_analysis_only": True}
            ),
            ResourceType.DIAGNOSTIC_RESULTS: AccessRule(
                ComponentRole.CLINICAL_PROCESSOR, ResourceType.DIAGNOSTIC_RESULTS,
                AccessLevel.READ_WRITE,
                conditions={"clinical_context": True}
            ),
            ResourceType.PATIENT_INFO: AccessRule(
                ComponentRole.CLINICAL_PROCESSOR, ResourceType.PATIENT_INFO,
                AccessLevel.READ_ONLY,
                conditions={"clinical_context_only": True, "encrypted_access": True}
            ),
            ResourceType.PROTOCOL_DATA: AccessRule(
                ComponentRole.CLINICAL_PROCESSOR, ResourceType.PROTOCOL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"clinical_protocols_only": True}
            )
        }
        
        # Medical Knowledge System - Base de conocimiento médico
        matrix[ComponentRole.MEDICAL_KNOWLEDGE] = {
            ResourceType.PROTOCOL_DATA: AccessRule(
                ComponentRole.MEDICAL_KNOWLEDGE, ResourceType.PROTOCOL_DATA,
                AccessLevel.READ_WRITE,
                conditions={"knowledge_base_management": True}
            ),
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.MEDICAL_KNOWLEDGE, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"anonymized_analysis_only": True, "no_patient_identification": True}
            ),
            ResourceType.DIAGNOSTIC_RESULTS: AccessRule(
                ComponentRole.MEDICAL_KNOWLEDGE, ResourceType.DIAGNOSTIC_RESULTS,
                AccessLevel.READ_ONLY,
                conditions={"pattern_analysis_only": True, "anonymized": True}
            )
        }
        
        # Human Review Queue - Escalamiento a personal médico
        matrix[ComponentRole.HUMAN_REVIEW_QUEUE] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.HUMAN_REVIEW_QUEUE, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_WRITE,
                conditions={"human_review_context": True, "qualified_personnel_only": True}
            ),
            ResourceType.PATIENT_INFO: AccessRule(
                ComponentRole.HUMAN_REVIEW_QUEUE, ResourceType.PATIENT_INFO,
                AccessLevel.READ_WRITE,
                conditions={"medical_review_only": True, "authorized_personnel": True}
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.HUMAN_REVIEW_QUEUE, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_WRITE,
                conditions={"medical_review_context": True}
            ),
            ResourceType.DIAGNOSTIC_RESULTS: AccessRule(
                ComponentRole.HUMAN_REVIEW_QUEUE, ResourceType.DIAGNOSTIC_RESULTS,
                AccessLevel.READ_WRITE,
                conditions={"human_validation": True}
            ),
            ResourceType.NOTIFICATION_DATA: AccessRule(
                ComponentRole.HUMAN_REVIEW_QUEUE, ResourceType.NOTIFICATION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"medical_notifications_only": True}
            )
        }
        
        # === CROSS-CUTTING SERVICES ===
        
        # Audit Service - Acceso completo para auditoría
        matrix[ComponentRole.AUDIT_SERVICE] = {
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.AUDIT_SERVICE, ResourceType.AUDIT_LOGS,
                AccessLevel.ADMIN,
                conditions={"audit_context": True}
            ),
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.AUDIT_SERVICE, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"audit_trail_only": True, "no_content_access": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.AUDIT_SERVICE, ResourceType.SESSION_DATA,
                AccessLevel.READ_ONLY,
                conditions={"audit_metadata_only": True}
            ),
            ResourceType.SYSTEM_CONFIG: AccessRule(
                ComponentRole.AUDIT_SERVICE, ResourceType.SYSTEM_CONFIG,
                AccessLevel.READ_ONLY,
                conditions={"compliance_monitoring": True}
            )
        }
        
        # Slack Orchestrator - Notificaciones médicas
        matrix[ComponentRole.SLACK_ORCHESTRATOR] = {
            ResourceType.NOTIFICATION_DATA: AccessRule(
                ComponentRole.SLACK_ORCHESTRATOR, ResourceType.NOTIFICATION_DATA,
                AccessLevel.READ_WRITE,
                conditions={"notification_context": True}
            ),
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.SLACK_ORCHESTRATOR, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_ONLY,
                conditions={"notification_formatting_only": True, "anonymized_summaries": True}
            ),
            ResourceType.DIAGNOSTIC_RESULTS: AccessRule(
                ComponentRole.SLACK_ORCHESTRATOR, ResourceType.DIAGNOSTIC_RESULTS,
                AccessLevel.READ_ONLY,
                conditions={"summary_notifications_only": True}
            ),
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.SLACK_ORCHESTRATOR, ResourceType.AUDIT_LOGS,
                AccessLevel.READ_WRITE,
                conditions={"notification_events_only": True}
            )
        }
        
        # === USUARIOS EXTERNOS ===
        
        # Medical Staff - Personal médico autorizado
        matrix[ComponentRole.MEDICAL_STAFF] = {
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.MEDICAL_STAFF, ResourceType.MEDICAL_DATA,
                AccessLevel.READ_WRITE,
                conditions={"authenticated_medical_personnel": True, "patient_care_context": True},
                time_restrictions={"business_hours_only": False, "max_session_duration": timedelta(hours=8)}
            ),
            ResourceType.PATIENT_INFO: AccessRule(
                ComponentRole.MEDICAL_STAFF, ResourceType.PATIENT_INFO,
                AccessLevel.READ_WRITE,
                conditions={"authorized_patient_access": True, "medical_necessity": True}
            ),
            ResourceType.CLINICAL_IMAGES: AccessRule(
                ComponentRole.MEDICAL_STAFF, ResourceType.CLINICAL_IMAGES,
                AccessLevel.READ_WRITE,
                conditions={"medical_analysis_context": True}
            ),
            ResourceType.DIAGNOSTIC_RESULTS: AccessRule(
                ComponentRole.MEDICAL_STAFF, ResourceType.DIAGNOSTIC_RESULTS,
                AccessLevel.READ_WRITE,
                conditions={"clinical_interpretation": True}
            )
        }
        
        # Admin User - Administrador del sistema
        matrix[ComponentRole.ADMIN_USER] = {
            ResourceType.SYSTEM_CONFIG: AccessRule(
                ComponentRole.ADMIN_USER, ResourceType.SYSTEM_CONFIG,
                AccessLevel.ADMIN,
                conditions={"authenticated_admin": True},
                time_restrictions={"require_mfa": True, "max_session_duration": timedelta(hours=2)}
            ),
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.ADMIN_USER, ResourceType.AUDIT_LOGS,
                AccessLevel.READ_ONLY,
                conditions={"administrative_review": True}
            ),
            ResourceType.ENCRYPTION_KEYS: AccessRule(
                ComponentRole.ADMIN_USER, ResourceType.ENCRYPTION_KEYS,
                AccessLevel.ADMIN,
                conditions={"key_management_context": True, "emergency_access_only": True}
            ),
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.ADMIN_USER, ResourceType.MEDICAL_DATA,
                AccessLevel.NO_ACCESS,
                audit_required=True  # Admins NO deben acceder a datos médicos
            )
        }
        
        # System Monitor - Monitoreo del sistema
        matrix[ComponentRole.SYSTEM_MONITOR] = {
            ResourceType.SYSTEM_CONFIG: AccessRule(
                ComponentRole.SYSTEM_MONITOR, ResourceType.SYSTEM_CONFIG,
                AccessLevel.READ_ONLY,
                conditions={"monitoring_context": True}
            ),
            ResourceType.AUDIT_LOGS: AccessRule(
                ComponentRole.SYSTEM_MONITOR, ResourceType.AUDIT_LOGS,
                AccessLevel.READ_ONLY,
                conditions={"system_health_only": True, "no_content_access": True}
            ),
            ResourceType.SESSION_DATA: AccessRule(
                ComponentRole.SYSTEM_MONITOR, ResourceType.SESSION_DATA,
                AccessLevel.READ_ONLY,
                conditions={"performance_metrics_only": True}
            ),
            ResourceType.MEDICAL_DATA: AccessRule(
                ComponentRole.SYSTEM_MONITOR, ResourceType.MEDICAL_DATA,
                AccessLevel.NO_ACCESS,
                audit_required=True
            )
        }
        
        return matrix
    
    async def check_access(self, request: AccessRequest) -> AccessDecision:
        """
        Verifica acceso basado en la matriz de control.
        
        Args:
            request: Solicitud de acceso con contexto
            
        Returns:
            Decisión de acceso con razón y auditoría
        """
        try:
            # Verificar cache primero
            cache_key = self._get_cache_key(request)
            if cache_key in self.decision_cache:
                cached_decision = self.decision_cache[cache_key]
                if self._is_cache_valid(cached_decision):
                    return cached_decision
            
            # Obtener regla de acceso
            access_rule = self._get_access_rule(request.component, request.resource)
            if not access_rule:
                decision = AccessDecision(
                    granted=False,
                    reason=f"No hay regla de acceso definida para {request.component.value} -> {request.resource.value}",
                    access_level=AccessLevel.NO_ACCESS
                )
            else:
                # Evaluar condiciones
                decision = await self._evaluate_access_conditions(request, access_rule)
            
            # Crear evento de auditoría si es requerido
            if access_rule and access_rule.audit_required:
                audit_event = await self._create_audit_event(request, decision)
                decision.audit_event = audit_event
                
                # Registrar en auditoría
                await self.audit_service.log_event(audit_event)
            
            # Cache de la decisión
            self.decision_cache[cache_key] = decision
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error verificando acceso: {str(e)}")
            
            # En caso de error, denegar acceso y auditar
            error_decision = AccessDecision(
                granted=False,
                reason=f"Error de sistema: {str(e)}",
                access_level=AccessLevel.NO_ACCESS
            )
            
            await self.audit_service.log_event(
                AuditEvent(
                    event_type="access_control_error",
                    session_id=request.session_id or "unknown",
                    level=AuditLevel.HIGH,
                    details={
                        "component": request.component.value,
                        "resource": request.resource.value,
                        "operation": request.operation,
                        "error": str(e)
                    }
                )
            )
            
            return error_decision
    
    def _get_access_rule(self, component: ComponentRole, resource: ResourceType) -> Optional[AccessRule]:
        """Obtiene la regla de acceso para un componente y recurso."""
        component_rules = self.access_matrix.get(component, {})
        return component_rules.get(resource)
    
    async def _evaluate_access_conditions(self, request: AccessRequest, rule: AccessRule) -> AccessDecision:
        """Evalúa las condiciones de acceso de una regla."""
        # Verificar nivel de acceso base
        if rule.access_level == AccessLevel.NO_ACCESS:
            return AccessDecision(
                granted=False,
                reason="Acceso explícitamente denegado por política",
                access_level=AccessLevel.NO_ACCESS
            )
        
        # Verificar operación permitida por nivel de acceso
        operation_allowed = self._check_operation_permission(request.operation, rule.access_level)
        if not operation_allowed:
            return AccessDecision(
                granted=False,
                reason=f"Operación '{request.operation}' no permitida para nivel {rule.access_level.name}",
                access_level=rule.access_level
            )
        
        # Evaluar condiciones específicas
        if rule.conditions:
            condition_check = await self._evaluate_conditions(request, rule.conditions)
            if not condition_check["passed"]:
                return AccessDecision(
                    granted=False,
                    reason=f"Condición falló: {condition_check['reason']}",
                    access_level=rule.access_level
                )
        
        # Verificar restricciones temporales
        if rule.time_restrictions:
            time_check = self._evaluate_time_restrictions(request, rule.time_restrictions)
            if not time_check["passed"]:
                return AccessDecision(
                    granted=False,
                    reason=f"Restricción temporal: {time_check['reason']}",
                    access_level=rule.access_level
                )
        
        # Verificar sesión activa si es requerida
        if request.session_id:
            session_check = await self._validate_session(request.session_id, rule)
            if not session_check["valid"]:
                return AccessDecision(
                    granted=False,
                    reason=f"Sesión inválida: {session_check['reason']}",
                    access_level=rule.access_level
                )
        
        # Acceso concedido
        return AccessDecision(
            granted=True,
            reason="Acceso concedido - todas las condiciones cumplidas",
            access_level=rule.access_level,
            conditions=rule.conditions
        )
    
    def _check_operation_permission(self, operation: str, access_level: AccessLevel) -> bool:
        """Verifica si una operación está permitida por el nivel de acceso."""
        operation_permissions = {
            AccessLevel.NO_ACCESS: [],
            AccessLevel.READ_ONLY: ["read", "list", "view"],
            AccessLevel.READ_WRITE: ["read", "list", "view", "write", "update", "create"],
            AccessLevel.ADMIN: ["read", "list", "view", "write", "update", "create", "delete", "admin"],
            AccessLevel.SYSTEM: ["*"]  # Todas las operaciones
        }
        
        allowed_ops = operation_permissions.get(access_level, [])
        return "*" in allowed_ops or operation.lower() in allowed_ops
    
    async def _evaluate_conditions(self, request: AccessRequest, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa condiciones específicas de acceso."""
        for condition, value in conditions.items():
            if condition == "session_scope_only":
                if not request.session_id:
                    return {"passed": False, "reason": "Se requiere session_id para acceso con scope de sesión"}
            
            elif condition == "no_medical_analysis":
                if request.context and request.context.get("medical_analysis_intent"):
                    return {"passed": False, "reason": "Análisis médico no permitido en esta capa"}
            
            elif condition == "encryption_required":
                if not (request.context and request.context.get("encrypted")):
                    return {"passed": False, "reason": "Se requiere encriptación para este recurso"}
            
            elif condition == "clinical_context":
                if not (request.context and request.context.get("clinical_context")):
                    return {"passed": False, "reason": "Se requiere contexto clínico válido"}
            
            elif condition == "authenticated_medical_personnel":
                if not await self._verify_medical_credentials(request):
                    return {"passed": False, "reason": "Credenciales médicas no válidas"}
            
            elif condition == "anonymized_analysis_only":
                if request.context and request.context.get("patient_identifiable"):
                    return {"passed": False, "reason": "Solo análisis anonimizado permitido"}
        
        return {"passed": True, "reason": "Todas las condiciones cumplidas"}
    
    def _evaluate_time_restrictions(self, request: AccessRequest, restrictions: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa restricciones temporales."""
        current_time = datetime.now()
        
        if "max_session_duration" in restrictions:
            max_duration = restrictions["max_session_duration"]
            if request.session_id and request.session_id in self.active_sessions:
                session_start = datetime.fromisoformat(self.active_sessions[request.session_id]["start_time"])
                if current_time - session_start > max_duration:
                    return {"passed": False, "reason": f"Sesión excede duración máxima de {max_duration}"}
        
        if "business_hours_only" in restrictions and restrictions["business_hours_only"]:
            if current_time.hour < 8 or current_time.hour > 18:
                return {"passed": False, "reason": "Acceso solo durante horario laboral (8:00-18:00)"}
        
        if "require_mfa" in restrictions and restrictions["require_mfa"]:
            if not (request.context and request.context.get("mfa_verified")):
                return {"passed": False, "reason": "Se requiere autenticación multifactor"}
        
        return {"passed": True, "reason": "Restricciones temporales cumplidas"}
    
    async def _validate_session(self, session_id: str, rule: AccessRule) -> Dict[str, Any]:
        """Valida sesión activa."""
        if session_id not in self.active_sessions:
            return {"valid": False, "reason": "Sesión no encontrada"}
        
        session = self.active_sessions[session_id]
        session_start = datetime.fromisoformat(session["start_time"])
        
        # Verificar expiración
        if datetime.now() - session_start > timedelta(minutes=15):  # Default timeout
            return {"valid": False, "reason": "Sesión expirada"}
        
        # Verificar límites de la sesión
        if session.get("access_count", 0) > session.get("max_access_count", 1000):
            return {"valid": False, "reason": "Límite de accesos de sesión excedido"}
        
        return {"valid": True, "reason": "Sesión válida"}
    
    async def _verify_medical_credentials(self, request: AccessRequest) -> bool:
        """Verifica credenciales médicas (implementación simulada)."""
        # En implementación real, verificaría contra base de datos de personal médico
        if not request.context:
            return False
        
        medical_id = request.context.get("medical_id")
        credentials = request.context.get("medical_credentials")
        
        return bool(medical_id and credentials)
    
    async def _create_audit_event(self, request: AccessRequest, decision: AccessDecision) -> AuditEvent:
        """Crea evento de auditoría para la decisión de acceso."""
        level = AuditLevel.HIGH if not decision.granted else AuditLevel.LOW
        
        return AuditEvent(
            event_type="access_control_decision",
            session_id=request.session_id or "no_session",
            level=level,
            details={
                "component": request.component.value,
                "resource": request.resource.value,
                "operation": request.operation,
                "access_granted": decision.granted,
                "access_level": decision.access_level.value,
                "reason": decision.reason,
                "timestamp": request.timestamp,
                "context": request.context or {}
            }
        )
    
    def _get_cache_key(self, request: AccessRequest) -> str:
        """Genera clave de cache para la solicitud."""
        return f"{request.component.value}:{request.resource.value}:{request.operation}:{request.session_id or 'no_session'}"
    
    def _is_cache_valid(self, decision: AccessDecision) -> bool:
        """Verifica si la decisión en cache sigue siendo válida."""
        # Cache simple por tiempo - en implementación real sería más sofisticado
        return True  # Por simplicidad, mantenemos cache activo
    
    async def register_session(self, session_id: str, component: ComponentRole, context: Dict[str, Any]) -> bool:
        """Registra una nueva sesión activa."""
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "component": component.value,
            "start_time": datetime.now().isoformat(),
            "context": context,
            "access_count": 0,
            "max_access_count": 1000
        }
        
        await self.audit_service.log_event(
            AuditEvent(
                event_type="session_registered",
                session_id=session_id,
                level=AuditLevel.LOW,
                details={
                    "component": component.value,
                    "context": context
                }
            )
        )
        
        return True
    
    async def unregister_session(self, session_id: str) -> bool:
        """Desregistra una sesión activa."""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            
            await self.audit_service.log_event(
                AuditEvent(
                    event_type="session_unregistered",
                    session_id=session_id,
                    level=AuditLevel.LOW,
                    details={
                        "component": session["component"],
                        "duration_seconds": (
                            datetime.now() - datetime.fromisoformat(session["start_time"])
                        ).total_seconds(),
                        "access_count": session["access_count"]
                    }
                )
            )
            return True
        return False
    
    def get_layer_permissions_summary(self) -> Dict[str, Dict[str, str]]:
        """Obtiene resumen de permisos por capa para documentación."""
        summary = {}
        
        for component, resources in self.access_matrix.items():
            component_summary = {}
            for resource, rule in resources.items():
                component_summary[resource.value] = {
                    "access_level": rule.access_level.name,
                    "conditions": len(rule.conditions or {}),
                    "audit_required": rule.audit_required
                }
            summary[component.value] = component_summary
        
        return summary