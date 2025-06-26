"""
Audit Service - Servicio Transversal de Auditoría
Sistema de auditoría completo para cumplimiento médico.

Responsabilidades:
- Registrar todas las transacciones médicas
- Mantener trazabilidad completa
- Cumplimiento HIPAA y regulatorio
- Alertas de seguridad
- Reportes de auditoría
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis.asyncio as redis

from .secure_logger import SecureLogger

logger = SecureLogger("audit_service")


class AuditEventType(Enum):
    """Tipos de eventos de auditoría."""
    # Eventos de acceso
    USER_ACCESS = "user_access"
    DATA_ACCESS = "data_access"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # Eventos de procesamiento
    IMAGE_PROCESSED = "image_processed"
    MEDICAL_DECISION = "medical_decision"
    TRIAGE_COMPLETED = "triage_completed"
    REVIEW_ASSIGNED = "review_assigned"
    
    # Eventos de datos
    DATA_CREATED = "data_created"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    
    # Eventos de seguridad
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_BREACH = "security_breach"
    
    # Eventos de sistema
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    CONFIGURATION_CHANGED = "configuration_changed"


class AuditSeverity(Enum):
    """Niveles de severidad de auditoría."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AuditLevel(Enum):
    """Niveles de auditoría (alias for AuditSeverity for compatibility)."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStandard(Enum):
    """Estándares de cumplimiento."""
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO13485 = "iso13485"
    FDA = "fda"
    GDPR = "gdpr"


@dataclass
class AuditEvent:
    """Evento de auditoría completo."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    component: str
    session_id: Optional[str]
    user_id: Optional[str]
    action: str
    resource: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    compliance_flags: List[ComplianceStandard]
    risk_score: float
    requires_alert: bool


@dataclass
class AuditTrail:
    """Cadena de auditoría para una sesión."""
    session_id: str
    created_at: datetime
    events: List[AuditEvent]
    total_events: int
    risk_assessment: str
    compliance_status: Dict[str, bool]
    summary: str


class AuditService:
    """
    Servicio de auditoría transversal para todo el sistema.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Inicializar servicio de auditoría.
        
        Args:
            redis_url: URL de Redis para persistencia
        """
        self.redis_url = redis_url or "redis://localhost:6379/3"  # DB dedicada para auditoría
        self.redis_client = None
        
        # Configuración de retención
        self.retention_config = {
            "critical_events": timedelta(days=2555),  # 7 años para HIPAA
            "high_events": timedelta(days=1095),      # 3 años
            "medium_events": timedelta(days=365),     # 1 año
            "low_events": timedelta(days=90)          # 90 días
        }
        
        # Configuración de alertas
        self.alert_thresholds = {
            "failed_authentications": 5,
            "permission_denials": 3,
            "suspicious_activities": 1,
            "high_risk_sessions": 1
        }
        
        # Patrones de riesgo
        self.risk_patterns = {
            "multiple_failed_logins": 0.8,
            "unusual_access_pattern": 0.6,
            "data_export_volume": 0.7,
            "off_hours_access": 0.4,
            "unauthorized_access_attempt": 0.9
        }
        
        # Mapeo de compliance por evento
        self.compliance_mapping = {
            AuditEventType.DATA_ACCESS: [ComplianceStandard.HIPAA, ComplianceStandard.GDPR],
            AuditEventType.IMAGE_PROCESSED: [ComplianceStandard.HIPAA, ComplianceStandard.ISO13485],
            AuditEventType.MEDICAL_DECISION: [ComplianceStandard.HIPAA, ComplianceStandard.FDA],
            AuditEventType.DATA_EXPORTED: [ComplianceStandard.HIPAA, ComplianceStandard.GDPR],
            AuditEventType.SECURITY_BREACH: [ComplianceStandard.HIPAA, ComplianceStandard.SOC2, ComplianceStandard.GDPR]
        }
        
        logger.audit("audit_service_initialized", {
            "component": "audit_service",
            "retention_configured": True,
            "compliance_standards": len(ComplianceStandard),
            "risk_patterns": len(self.risk_patterns)
        })
    
    async def initialize(self):
        """Inicializar servicio de auditoría."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Iniciar tareas de mantenimiento
            asyncio.create_task(self._audit_retention_cleanup())
            asyncio.create_task(self._risk_monitoring())
            asyncio.create_task(self._compliance_monitoring())
            
            # Registrar inicio del sistema
            await self.log_event(
                event_type=AuditEventType.SYSTEM_START,
                component="audit_service",
                action="service_initialized",
                details={"initialization_time": datetime.now(timezone.utc).isoformat()}
            )
            
            logger.audit("audit_service_ready", {
                "redis_connected": True,
                "monitoring_active": True
            })
            
        except Exception as e:
            logger.error("audit_service_init_failed", {"error": str(e)})
            raise
    
    async def log_event(self,
                       event_type: AuditEventType,
                       component: str,
                       action: str,
                       details: Optional[Dict[str, Any]] = None,
                       session_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       resource: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> str:
        """
        Registrar evento de auditoría.
        
        Args:
            event_type: Tipo de evento
            component: Componente que genera el evento
            action: Acción realizada
            details: Detalles adicionales
            session_id: ID de sesión (opcional)
            user_id: ID de usuario (opcional)
            resource: Recurso afectado (opcional)
            ip_address: Dirección IP (opcional)
            user_agent: User agent (opcional)
            
        Returns:
            ID del evento de auditoría
        """
        try:
            # Generar ID único del evento
            event_id = self._generate_event_id()
            
            # Determinar severidad
            severity = self._determine_severity(event_type, details)
            
            # Determinar estándares de compliance aplicables
            compliance_flags = self.compliance_mapping.get(event_type, [])
            
            # Calcular score de riesgo
            risk_score = self._calculate_risk_score(event_type, details, user_id, ip_address)
            
            # Determinar si requiere alerta
            requires_alert = self._requires_alert(event_type, severity, risk_score)
            
            # Crear evento
            audit_event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(timezone.utc),
                component=component,
                session_id=session_id,
                user_id=user_id,
                action=action,
                resource=resource,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                compliance_flags=compliance_flags,
                risk_score=risk_score,
                requires_alert=requires_alert
            )
            
            # Guardar en Redis
            await self._save_audit_event(audit_event)
            
            # Actualizar trail de sesión si aplica
            if session_id:
                await self._update_session_trail(session_id, audit_event)
            
            # Enviar alerta si es necesario
            if requires_alert:
                await self._send_alert(audit_event)
            
            # Log interno (sin PII)
            logger.audit("audit_event_logged", {
                "event_id": event_id,
                "event_type": event_type.value,
                "component": component,
                "action": action,
                "severity": severity.value,
                "risk_score": risk_score,
                "requires_alert": requires_alert
            })
            
            return event_id
            
        except Exception as e:
            logger.error("audit_logging_failed", {
                "event_type": event_type.value,
                "component": component,
                "error": str(e)
            })
            # En caso de falla, usar logging como fallback
            logger.audit(f"fallback_{event_type.value}", {
                "component": component,
                "action": action,
                "details": details,
                "error": "audit_service_unavailable"
            })
            return "fallback_event"
    
    async def get_session_trail(self, session_id: str) -> Optional[AuditTrail]:
        """
        Obtener trail completo de auditoría para una sesión.
        
        Args:
            session_id: ID de sesión
            
        Returns:
            AuditTrail o None si no existe
        """
        try:
            # Obtener eventos de la sesión
            events = await self._get_session_events(session_id)
            
            if not events:
                return None
            
            # Crear trail
            first_event = min(events, key=lambda e: e.timestamp)
            
            # Evaluar riesgo de la sesión
            risk_assessment = self._assess_session_risk(events)
            
            # Verificar compliance
            compliance_status = self._check_session_compliance(events)
            
            # Generar resumen
            summary = self._generate_session_summary(events)
            
            return AuditTrail(
                session_id=session_id,
                created_at=first_event.timestamp,
                events=events,
                total_events=len(events),
                risk_assessment=risk_assessment,
                compliance_status=compliance_status,
                summary=summary
            )
            
        except Exception as e:
            logger.error("get_session_trail_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return None
    
    async def generate_compliance_report(self, 
                                       standard: ComplianceStandard,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """
        Generar reporte de cumplimiento.
        
        Args:
            standard: Estándar de compliance
            start_date: Fecha de inicio
            end_date: Fecha final
            
        Returns:
            Dict con reporte de compliance
        """
        try:
            # Obtener eventos relevantes
            events = await self._get_events_by_compliance(standard, start_date, end_date)
            
            # Analizar eventos
            analysis = {
                "total_events": len(events),
                "by_type": {},
                "by_severity": {},
                "violations": [],
                "risk_events": [],
                "compliance_score": 0.0
            }
            
            violations = 0
            high_risk_events = 0
            
            for event in events:
                # Contar por tipo
                event_type = event.event_type.value
                if event_type not in analysis["by_type"]:
                    analysis["by_type"][event_type] = 0
                analysis["by_type"][event_type] += 1
                
                # Contar por severidad
                severity = event.severity.value
                if severity not in analysis["by_severity"]:
                    analysis["by_severity"][severity] = 0
                analysis["by_severity"][severity] += 1
                
                # Detectar violaciones
                if self._is_compliance_violation(event, standard):
                    violations += 1
                    analysis["violations"].append({
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "violation_type": self._get_violation_type(event, standard),
                        "severity": event.severity.value
                    })
                
                # Eventos de alto riesgo
                if event.risk_score > 0.7:
                    high_risk_events += 1
                    analysis["risk_events"].append({
                        "event_id": event.event_id,
                        "risk_score": event.risk_score,
                        "event_type": event.event_type.value
                    })
            
            # Calcular score de compliance
            if len(events) > 0:
                analysis["compliance_score"] = max(0.0, 1.0 - (violations / len(events)))
            else:
                analysis["compliance_score"] = 1.0
            
            # Crear reporte
            report = {
                "standard": standard.value,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "analysis": analysis,
                "summary": {
                    "total_events": len(events),
                    "violations_count": violations,
                    "high_risk_events": high_risk_events,
                    "compliance_score": analysis["compliance_score"],
                    "status": "compliant" if violations == 0 else "non_compliant"
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "recommendations": self._generate_compliance_recommendations(analysis, standard)
            }
            
            logger.audit("compliance_report_generated", {
                "standard": standard.value,
                "total_events": len(events),
                "violations": violations,
                "compliance_score": analysis["compliance_score"]
            })
            
            return report
            
        except Exception as e:
            logger.error("compliance_report_failed", {
                "standard": standard.value,
                "error": str(e)
            })
            return {
                "error": str(e),
                "standard": standard.value,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def search_events(self,
                          filters: Dict[str, Any],
                          limit: int = 100) -> List[AuditEvent]:
        """
        Buscar eventos de auditoría con filtros.
        
        Args:
            filters: Filtros de búsqueda
            limit: Límite de resultados
            
        Returns:
            Lista de eventos de auditoría
        """
        try:
            # Obtener todas las claves de eventos en el rango de tiempo
            start_time = filters.get("start_time")
            end_time = filters.get("end_time", datetime.now(timezone.utc))
            
            # Buscar por patrón en Redis
            pattern = "audit_event:*"
            event_keys = await self.redis_client.keys(pattern)
            
            events = []
            for key in event_keys:
                if len(events) >= limit:
                    break
                
                event_data = await self.redis_client.hgetall(key)
                if event_data:
                    event = self._deserialize_event(event_data)
                    
                    # Aplicar filtros
                    if self._matches_filters(event, filters):
                        events.append(event)
            
            # Ordenar por timestamp
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return events[:limit]
            
        except Exception as e:
            logger.error("event_search_failed", {
                "filters": filters,
                "error": str(e)
            })
            return []
    
    def _generate_event_id(self) -> str:
        """Generar ID único para evento."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"AUD_{timestamp}_{unique_id}"
    
    def _determine_severity(self, 
                          event_type: AuditEventType, 
                          details: Optional[Dict[str, Any]]) -> AuditSeverity:
        """Determinar severidad del evento."""
        # Eventos críticos
        critical_events = [
            AuditEventType.SECURITY_BREACH,
            AuditEventType.UNAUTHORIZED_ACCESS,
            AuditEventType.DATA_DELETED
        ]
        
        if event_type in critical_events:
            return AuditSeverity.CRITICAL
        
        # Eventos de alta severidad
        high_events = [
            AuditEventType.AUTHENTICATION_FAILED,
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.SUSPICIOUS_ACTIVITY,
            AuditEventType.DATA_EXPORTED
        ]
        
        if event_type in high_events:
            return AuditSeverity.HIGH
        
        # Eventos médicos siempre son de severidad media o alta
        medical_events = [
            AuditEventType.IMAGE_PROCESSED,
            AuditEventType.MEDICAL_DECISION,
            AuditEventType.TRIAGE_COMPLETED
        ]
        
        if event_type in medical_events:
            return AuditSeverity.MEDIUM
        
        # Verificar detalles para ajustar severidad
        if details:
            if details.get("error") or details.get("failed"):
                return AuditSeverity.HIGH
            if details.get("emergency") or details.get("critical"):
                return AuditSeverity.CRITICAL
        
        return AuditSeverity.LOW
    
    def _calculate_risk_score(self,
                            event_type: AuditEventType,
                            details: Optional[Dict[str, Any]],
                            user_id: Optional[str],
                            ip_address: Optional[str]) -> float:
        """Calcular score de riesgo del evento."""
        base_risk = 0.1
        
        # Riesgo base por tipo de evento
        risk_by_type = {
            AuditEventType.SECURITY_BREACH: 1.0,
            AuditEventType.UNAUTHORIZED_ACCESS: 0.9,
            AuditEventType.AUTHENTICATION_FAILED: 0.7,
            AuditEventType.SUSPICIOUS_ACTIVITY: 0.8,
            AuditEventType.DATA_EXPORTED: 0.6,
            AuditEventType.PERMISSION_DENIED: 0.5,
            AuditEventType.MEDICAL_DECISION: 0.4,
            AuditEventType.DATA_MODIFIED: 0.3
        }
        
        base_risk = risk_by_type.get(event_type, 0.1)
        
        # Ajustes por detalles
        if details:
            if details.get("failed_attempts", 0) > 3:
                base_risk += 0.2
            if details.get("emergency"):
                base_risk += 0.3
            if details.get("large_data_volume"):
                base_risk += 0.2
            if details.get("off_hours"):
                base_risk += 0.1
        
        # Ajustes por patrón de usuario (requeriría historial)
        # En implementación completa, verificaría patrones históricos
        
        return min(1.0, base_risk)
    
    def _requires_alert(self,
                       event_type: AuditEventType,
                       severity: AuditSeverity,
                       risk_score: float) -> bool:
        """Determinar si el evento requiere alerta."""
        # Siempre alertar eventos críticos
        if severity == AuditSeverity.CRITICAL:
            return True
        
        # Alertar eventos de alto riesgo
        if risk_score > 0.7:
            return True
        
        # Alertar eventos específicos
        alert_events = [
            AuditEventType.SECURITY_BREACH,
            AuditEventType.UNAUTHORIZED_ACCESS,
            AuditEventType.SUSPICIOUS_ACTIVITY
        ]
        
        return event_type in alert_events
    
    async def _save_audit_event(self, event: AuditEvent):
        """Guardar evento en Redis."""
        key = f"audit_event:{event.event_id}"
        
        # Serializar evento
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "timestamp": event.timestamp.isoformat(),
            "component": event.component,
            "session_id": event.session_id or "",
            "user_id": event.user_id or "",
            "action": event.action,
            "resource": event.resource or "",
            "details": json.dumps(event.details),
            "ip_address": event.ip_address or "",
            "user_agent": event.user_agent or "",
            "compliance_flags": json.dumps([f.value for f in event.compliance_flags]),
            "risk_score": event.risk_score,
            "requires_alert": str(event.requires_alert)
        }
        
        # Guardar en Redis
        await self.redis_client.hset(key, mapping=data)
        
        # Set TTL basado en severidad
        ttl_config = {
            AuditSeverity.CRITICAL: self.retention_config["critical_events"],
            AuditSeverity.HIGH: self.retention_config["high_events"],
            AuditSeverity.MEDIUM: self.retention_config["medium_events"],
            AuditSeverity.LOW: self.retention_config["low_events"]
        }
        
        ttl_seconds = int(ttl_config[event.severity].total_seconds())
        await self.redis_client.expire(key, ttl_seconds)
        
        # Añadir a índices para búsqueda
        await self._update_search_indices(event)
    
    async def _update_search_indices(self, event: AuditEvent):
        """Actualizar índices de búsqueda."""
        timestamp_key = event.timestamp.strftime("%Y%m%d")
        
        # Índice por fecha
        await self.redis_client.sadd(f"audit_index:date:{timestamp_key}", event.event_id)
        await self.redis_client.expire(f"audit_index:date:{timestamp_key}", 
                                     int(self.retention_config["critical_events"].total_seconds()))
        
        # Índice por tipo de evento
        await self.redis_client.sadd(f"audit_index:type:{event.event_type.value}", event.event_id)
        
        # Índice por sesión
        if event.session_id:
            await self.redis_client.sadd(f"audit_index:session:{event.session_id}", event.event_id)
        
        # Índice por componente
        await self.redis_client.sadd(f"audit_index:component:{event.component}", event.event_id)
    
    async def _update_session_trail(self, session_id: str, event: AuditEvent):
        """Actualizar trail de sesión."""
        trail_key = f"audit_trail:{session_id}"
        
        # Añadir evento al trail
        await self.redis_client.lpush(trail_key, event.event_id)
        
        # Mantener solo últimos 100 eventos por sesión
        await self.redis_client.ltrim(trail_key, 0, 99)
        
        # Set TTL para el trail
        await self.redis_client.expire(trail_key, 
                                     int(self.retention_config["high_events"].total_seconds()))
    
    async def _send_alert(self, event: AuditEvent):
        """Enviar alerta para evento crítico."""
        try:
            # En implementación completa, esto integraría con:
            # - Sistema de notificaciones (email, SMS, Slack)
            # - SIEM (Security Information and Event Management)
            # - Monitoring tools (Grafana, DataDog)
            
            alert_data = {
                "alert_id": f"ALERT_{event.event_id}",
                "event_id": event.event_id,
                "severity": event.severity.value,
                "event_type": event.event_type.value,
                "risk_score": event.risk_score,
                "timestamp": event.timestamp.isoformat(),
                "component": event.component,
                "action": event.action,
                "requires_immediate_attention": event.severity == AuditSeverity.CRITICAL
            }
            
            # Guardar alerta
            alert_key = f"audit_alert:{alert_data['alert_id']}"
            await self.redis_client.hset(alert_key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in alert_data.items()
            })
            
            # TTL de 30 días para alertas
            await self.redis_client.expire(alert_key, 2592000)
            
            logger.audit("audit_alert_sent", {
                "alert_id": alert_data["alert_id"],
                "event_id": event.event_id,
                "severity": event.severity.value
            })
            
        except Exception as e:
            logger.error("alert_send_failed", {
                "event_id": event.event_id,
                "error": str(e)
            })
    
    async def _get_session_events(self, session_id: str) -> List[AuditEvent]:
        """Obtener eventos de una sesión."""
        try:
            trail_key = f"audit_trail:{session_id}"
            event_ids = await self.redis_client.lrange(trail_key, 0, -1)
            
            events = []
            for event_id in event_ids:
                event_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id
                event_key = f"audit_event:{event_id_str}"
                event_data = await self.redis_client.hgetall(event_key)
                
                if event_data:
                    event = self._deserialize_event(event_data)
                    events.append(event)
            
            # Ordenar por timestamp
            events.sort(key=lambda e: e.timestamp)
            return events
            
        except Exception as e:
            logger.error("get_session_events_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return []
    
    def _deserialize_event(self, event_data: Dict[str, Any]) -> AuditEvent:
        """Deserializar evento desde Redis."""
        return AuditEvent(
            event_id=event_data["event_id"],
            event_type=AuditEventType(event_data["event_type"]),
            severity=AuditSeverity(int(event_data["severity"])),
            timestamp=datetime.fromisoformat(event_data["timestamp"]),
            component=event_data["component"],
            session_id=event_data["session_id"] or None,
            user_id=event_data["user_id"] or None,
            action=event_data["action"],
            resource=event_data["resource"] or None,
            details=json.loads(event_data["details"]),
            ip_address=event_data["ip_address"] or None,
            user_agent=event_data["user_agent"] or None,
            compliance_flags=[ComplianceStandard(f) for f in json.loads(event_data["compliance_flags"])],
            risk_score=float(event_data["risk_score"]),
            requires_alert=event_data["requires_alert"] == "True"
        )
    
    def _assess_session_risk(self, events: List[AuditEvent]) -> str:
        """Evaluar riesgo de sesión."""
        if not events:
            return "unknown"
        
        avg_risk = sum(e.risk_score for e in events) / len(events)
        max_risk = max(e.risk_score for e in events)
        
        if max_risk >= 0.9 or avg_risk >= 0.7:
            return "high"
        elif max_risk >= 0.6 or avg_risk >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _check_session_compliance(self, events: List[AuditEvent]) -> Dict[str, bool]:
        """Verificar compliance de sesión."""
        compliance_status = {}
        
        for standard in ComplianceStandard:
            # Verificar si hay violaciones para este estándar
            violations = [e for e in events if self._is_compliance_violation(e, standard)]
            compliance_status[standard.value] = len(violations) == 0
        
        return compliance_status
    
    def _generate_session_summary(self, events: List[AuditEvent]) -> str:
        """Generar resumen de sesión."""
        if not events:
            return "No events recorded"
        
        event_types = set(e.event_type.value for e in events)
        components = set(e.component for e in events)
        duration = (events[-1].timestamp - events[0].timestamp).total_seconds() / 60
        
        summary = f"Session with {len(events)} events across {len(components)} components. "
        summary += f"Duration: {duration:.1f} minutes. "
        summary += f"Event types: {', '.join(event_types)}."
        
        return summary
    
    async def _get_events_by_compliance(self,
                                      standard: ComplianceStandard,
                                      start_date: datetime,
                                      end_date: datetime) -> List[AuditEvent]:
        """Obtener eventos relevantes para un estándar de compliance."""
        events = []
        
        # Buscar en índices por fecha
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime("%Y%m%d")
            index_key = f"audit_index:date:{date_key}"
            
            event_ids = await self.redis_client.smembers(index_key)
            
            for event_id in event_ids:
                event_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id
                event_key = f"audit_event:{event_id_str}"
                event_data = await self.redis_client.hgetall(event_key)
                
                if event_data:
                    event = self._deserialize_event(event_data)
                    
                    # Verificar si es relevante para el estándar
                    if standard in event.compliance_flags:
                        events.append(event)
            
            current_date += timedelta(days=1)
        
        return events
    
    def _is_compliance_violation(self, event: AuditEvent, standard: ComplianceStandard) -> bool:
        """Verificar si el evento constituye una violación de compliance."""
        # Definir reglas de violación por estándar
        violation_rules = {
            ComplianceStandard.HIPAA: [
                AuditEventType.UNAUTHORIZED_ACCESS,
                AuditEventType.DATA_EXPORTED,
                AuditEventType.SECURITY_BREACH
            ],
            ComplianceStandard.GDPR: [
                AuditEventType.UNAUTHORIZED_ACCESS,
                AuditEventType.DATA_EXPORTED
            ],
            ComplianceStandard.SOC2: [
                AuditEventType.SECURITY_BREACH,
                AuditEventType.UNAUTHORIZED_ACCESS
            ]
        }
        
        violation_events = violation_rules.get(standard, [])
        return event.event_type in violation_events
    
    def _get_violation_type(self, event: AuditEvent, standard: ComplianceStandard) -> str:
        """Obtener tipo de violación específica."""
        violation_types = {
            (ComplianceStandard.HIPAA, AuditEventType.UNAUTHORIZED_ACCESS): "unauthorized_phi_access",
            (ComplianceStandard.HIPAA, AuditEventType.DATA_EXPORTED): "phi_disclosure",
            (ComplianceStandard.GDPR, AuditEventType.DATA_EXPORTED): "personal_data_breach",
            (ComplianceStandard.SOC2, AuditEventType.SECURITY_BREACH): "security_control_failure"
        }
        
        return violation_types.get((standard, event.event_type), "general_violation")
    
    def _generate_compliance_recommendations(self, 
                                           analysis: Dict[str, Any],
                                           standard: ComplianceStandard) -> List[str]:
        """Generar recomendaciones de compliance."""
        recommendations = []
        
        if analysis["violations_count"] > 0:
            recommendations.append("Investigar y remediar violaciones identificadas")
            recommendations.append("Revisar controles de acceso y permisos")
        
        if analysis["high_risk_events"] > 0:
            recommendations.append("Implementar monitoreo adicional para eventos de alto riesgo")
        
        if analysis["compliance_score"] < 0.9:
            recommendations.append("Fortalecer procesos de compliance")
            recommendations.append("Capacitar personal en requisitos regulatorios")
        
        # Recomendaciones específicas por estándar
        if standard == ComplianceStandard.HIPAA:
            recommendations.extend([
                "Verificar cifrado de datos PHI",
                "Revisar logs de acceso a información médica"
            ])
        elif standard == ComplianceStandard.GDPR:
            recommendations.extend([
                "Verificar consentimientos de procesamiento",
                "Revisar retención de datos personales"
            ])
        
        return recommendations
    
    def _matches_filters(self, event: AuditEvent, filters: Dict[str, Any]) -> bool:
        """Verificar si evento coincide con filtros."""
        # Filtro por tiempo
        if "start_time" in filters:
            if event.timestamp < filters["start_time"]:
                return False
        
        if "end_time" in filters:
            if event.timestamp > filters["end_time"]:
                return False
        
        # Filtro por tipo de evento
        if "event_type" in filters:
            if event.event_type.value != filters["event_type"]:
                return False
        
        # Filtro por severidad
        if "severity" in filters:
            if event.severity.value != filters["severity"]:
                return False
        
        # Filtro por componente
        if "component" in filters:
            if event.component != filters["component"]:
                return False
        
        # Filtro por sesión
        if "session_id" in filters:
            if event.session_id != filters["session_id"]:
                return False
        
        # Filtro por usuario
        if "user_id" in filters:
            if event.user_id != filters["user_id"]:
                return False
        
        return True
    
    async def _audit_retention_cleanup(self):
        """Tarea de limpieza de retención de auditoría."""
        while True:
            try:
                await asyncio.sleep(86400)  # Ejecutar diariamente
                
                # Limpiar eventos expirados (Redis lo hace automáticamente con TTL)
                # Pero podemos hacer limpieza adicional de índices
                
                current_date = datetime.now(timezone.utc)
                cutoff_date = current_date - self.retention_config["low_events"]
                
                # Limpiar índices de fechas antiguas
                cleanup_date = cutoff_date
                while cleanup_date < current_date - timedelta(days=1):
                    date_key = cleanup_date.strftime("%Y%m%d")
                    index_key = f"audit_index:date:{date_key}"
                    
                    # Verificar si el índice está vacío
                    count = await self.redis_client.scard(index_key)
                    if count == 0:
                        await self.redis_client.delete(index_key)
                    
                    cleanup_date += timedelta(days=1)
                
                logger.audit("audit_retention_cleanup_completed", {
                    "cleanup_date": current_date.isoformat(),
                    "cutoff_date": cutoff_date.isoformat()
                })
                
            except Exception as e:
                logger.error("audit_retention_cleanup_failed", {"error": str(e)})
                await asyncio.sleep(3600)  # Retry en 1 hora
    
    async def _risk_monitoring(self):
        """Monitoreo continuo de riesgo."""
        while True:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                # Buscar patrones de riesgo en eventos recientes
                recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
                
                # Obtener eventos recientes de alto riesgo
                filters = {
                    "start_time": recent_time,
                    "end_time": datetime.now(timezone.utc)
                }
                
                recent_events = await self.search_events(filters, limit=1000)
                high_risk_events = [e for e in recent_events if e.risk_score > 0.7]
                
                if len(high_risk_events) > 5:  # Umbral configurable
                    await self.log_event(
                        event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                        component="audit_service",
                        action="high_risk_pattern_detected",
                        details={
                            "high_risk_count": len(high_risk_events),
                            "time_window": "1_hour",
                            "requires_investigation": True
                        }
                    )
                
            except Exception as e:
                logger.error("risk_monitoring_failed", {"error": str(e)})
                await asyncio.sleep(900)  # Retry en 15 minutos
    
    async def _compliance_monitoring(self):
        """Monitoreo continuo de compliance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cada hora
                
                # Verificar compliance para cada estándar
                for standard in ComplianceStandard:
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(hours=24)
                    
                    report = await self.generate_compliance_report(
                        standard, start_time, end_time
                    )
                    
                    if report.get("summary", {}).get("status") == "non_compliant":
                        await self.log_event(
                            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                            component="audit_service",
                            action="compliance_violation_detected",
                            details={
                                "standard": standard.value,
                                "violations": report.get("summary", {}).get("violations_count", 0),
                                "compliance_score": report.get("summary", {}).get("compliance_score", 0)
                            }
                        )
                
            except Exception as e:
                logger.error("compliance_monitoring_failed", {"error": str(e)})
                await asyncio.sleep(1800)  # Retry en 30 minutos


# Singleton global para el servicio de auditoría
_audit_service_instance = None

async def get_audit_service() -> AuditService:
    """Obtener instancia singleton del servicio de auditoría."""
    global _audit_service_instance
    
    if _audit_service_instance is None:
        _audit_service_instance = AuditService()
        await _audit_service_instance.initialize()
    
    return _audit_service_instance


async def audit_log(event_type: AuditEventType,
                   component: str,
                   action: str,
                   **kwargs) -> str:
    """Función de conveniencia para logging de auditoría."""
    audit_service = await get_audit_service()
    return await audit_service.log_event(
        event_type=event_type,
        component=component,
        action=action,
        **kwargs
    )