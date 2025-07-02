"""
Human Review Queue - Capa 3: Sistema de Revisión Humana
Sistema de escalamiento con revisión médica profesional.

Responsabilidades:
- Gestionar cola de casos para revisión humana
- Escalamiento automático por urgencia
- Notificaciones a personal médico
- Tracking de tiempos de respuesta
- Asegurar human-in-the-loop para casos críticos
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import redis.asyncio as redis

from ..core.input_packager import StandardizedInput
from ..core.medical_dispatcher import TriageDecision
from ..core.session_manager import SessionManager, SessionState
from ..messaging.slack_notifier_refactored import SlackNotifier
from ..utils.secure_logger import SecureLogger

logger = SecureLogger("human_review_queue")


class ReviewPriority(Enum):
    """Prioridades de revisión humana."""
    EMERGENCY = 1    # Respuesta inmediata
    URGENT = 2       # < 30 minutos
    HIGH = 3         # < 2 horas
    MEDIUM = 4       # < 8 horas
    LOW = 5          # < 24 horas


class ReviewStatus(Enum):
    """Estados de revisión."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ReviewerRole(Enum):
    """Roles de revisores médicos."""
    ATTENDING_PHYSICIAN = "attending_physician"
    NURSE_SPECIALIST = "nurse_specialist"
    WOUND_CARE_SPECIALIST = "wound_care_specialist"
    MEDICAL_RESIDENT = "medical_resident"
    CLINICAL_SUPERVISOR = "clinical_supervisor"


@dataclass
class ReviewItem:
    """Item para revisión humana."""
    review_id: str
    session_id: str
    priority: ReviewPriority
    status: ReviewStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    escalation_count: int = 0
    last_escalated: Optional[datetime] = None
    required_role: Optional[ReviewerRole] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ReviewResult:
    """Resultado de revisión humana."""
    review_id: str
    session_id: str
    reviewer_id: str
    decision: str
    clinical_notes: str
    recommendations: List[str]
    requires_follow_up: bool
    follow_up_timeline: Optional[str]
    reviewed_at: datetime
    review_time_minutes: float


class HumanReviewQueue:
    """
    Sistema de cola de revisión humana con escalamiento automático.
    """
    
    def __init__(self, 
                 redis_url: Optional[str] = None,
                 slack_notifier: Optional[SlackNotifier] = None,
                 session_manager: Optional[SessionManager] = None):
        """
        Inicializar sistema de revisión humana.
        
        Args:
            redis_url: URL de Redis para persistencia
            slack_notifier: Notificador de Slack
            session_manager: Gestor de sesiones
        """
        self.redis_url = redis_url or "redis://localhost:6379/2"  # DB separada
        self.redis_client = None
        self.slack_notifier = slack_notifier
        self.session_manager = session_manager
        
        # Configuración de tiempos de respuesta por prioridad
        self.response_times = {
            ReviewPriority.EMERGENCY: timedelta(minutes=5),
            ReviewPriority.URGENT: timedelta(minutes=30),
            ReviewPriority.HIGH: timedelta(hours=2),
            ReviewPriority.MEDIUM: timedelta(hours=8),
            ReviewPriority.LOW: timedelta(hours=24)
        }
        
        # Configuración de escalamiento
        self.escalation_intervals = {
            ReviewPriority.EMERGENCY: timedelta(minutes=2),
            ReviewPriority.URGENT: timedelta(minutes=10),
            ReviewPriority.HIGH: timedelta(minutes=30),
            ReviewPriority.MEDIUM: timedelta(hours=2),
            ReviewPriority.LOW: timedelta(hours=4)
        }
        
        # Roles requeridos por tipo de caso
        self.role_requirements = {
            "lpp_grade_3_4": ReviewerRole.WOUND_CARE_SPECIALIST,
            "infection_risk": ReviewerRole.ATTENDING_PHYSICIAN,
            "medication_query": ReviewerRole.ATTENDING_PHYSICIAN,
            "emergency": ReviewerRole.ATTENDING_PHYSICIAN,
            "general": ReviewerRole.NURSE_SPECIALIST
        }
        
        # Personal médico disponible (en producción vendría de LDAP/AD)
        self.medical_staff = self._initialize_medical_staff()
        
        logger.audit("human_review_queue_initialized", {
            "component": "layer3_human_review",
            "response_times_configured": True,
            "escalation_enabled": True,
            "medical_staff_count": len(self.medical_staff)
        })
    
    async def initialize(self):
        """Inicializar servicios."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Iniciar tareas de monitoreo
            asyncio.create_task(self._monitor_queue_timeouts())
            asyncio.create_task(self._escalation_monitor())
            
            logger.audit("human_review_queue_ready", {
                "redis_connected": True,
                "monitoring_active": True
            })
            
        except Exception as e:
            logger.error("human_review_queue_init_failed", {
                "error": str(e)
            })
            raise
    
    async def process(self, 
                     standardized_input: StandardizedInput,
                     triage_decision: TriageDecision) -> Dict[str, Any]:
        """
        Procesar caso para revisión humana.
        
        Args:
            standardized_input: Input estandarizado
            triage_decision: Decisión del triage
            
        Returns:
            Dict con información de encolamiento
        """
        session_id = standardized_input.session_id
        
        try:
            # Determinar prioridad de revisión
            priority = self._determine_review_priority(triage_decision, standardized_input)
            
            # Determinar rol requerido
            required_role = self._determine_required_role(triage_decision, standardized_input)
            
            # Crear item de revisión
            review_item = await self._create_review_item(
                session_id,
                priority,
                required_role,
                triage_decision,
                standardized_input
            )
            
            # Encolar para revisión
            queue_result = await self._enqueue_for_review(review_item)
            
            # Notificar al personal médico
            await self._notify_medical_staff(review_item)
            
            # Actualizar estado de sesión
            if self.session_manager:
                await self.session_manager.update_session_state(
                    session_id,
                    SessionState.REVIEW_PENDING,
                    additional_data={
                        "review_id": review_item.review_id,
                        "priority": priority.name,
                        "required_role": required_role.value if required_role else None
                    }
                )
            
            logger.audit("case_queued_for_review", {
                "session_id": session_id,
                "review_id": review_item.review_id,
                "priority": priority.name,
                "required_role": required_role.value if required_role else None,
                "queue_position": queue_result.get("position", 0)
            })
            
            return {
                "success": True,
                "review_id": review_item.review_id,
                "priority": priority.name,
                "estimated_wait_time": self._estimate_wait_time(priority),
                "queue_position": queue_result.get("position", 0),
                "assigned_staff": queue_result.get("assigned_staff"),
                "next_escalation": self._calculate_next_escalation(review_item)
            }
            
        except Exception as e:
            logger.error("human_review_processing_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e),
                "fallback_contact": "emergency_medical_line"
            }
    
    async def assign_reviewer(self, review_id: str, reviewer_id: str) -> Dict[str, Any]:
        """
        Asignar revisor a un caso.
        
        Args:
            review_id: ID de revisión
            reviewer_id: ID del revisor
            
        Returns:
            Dict con resultado de asignación
        """
        try:
            # Obtener item de revisión
            review_item = await self._get_review_item(review_id)
            if not review_item:
                return {
                    "success": False,
                    "error": "Review item not found"
                }
            
            # Verificar que el revisor tiene el rol adecuado
            reviewer = self.medical_staff.get(reviewer_id)
            if not reviewer:
                return {
                    "success": False,
                    "error": "Reviewer not found"
                }
            
            if review_item.required_role and reviewer["role"] != review_item.required_role:
                return {
                    "success": False,
                    "error": f"Reviewer role {reviewer['role'].value} insufficient for required {review_item.required_role.value}"
                }
            
            # Asignar
            review_item.assigned_to = reviewer_id
            review_item.assigned_at = datetime.now(timezone.utc)
            review_item.status = ReviewStatus.ASSIGNED
            
            # Guardar en Redis
            await self._save_review_item(review_item)
            
            # Notificar asignación
            await self._notify_assignment(review_item, reviewer)
            
            logger.audit("review_assigned", {
                "review_id": review_id,
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer["role"].value,
                "assigned_at": review_item.assigned_at.isoformat()
            })
            
            return {
                "success": True,
                "assigned_to": reviewer["name"],
                "assigned_at": review_item.assigned_at.isoformat(),
                "expected_completion": self._calculate_expected_completion(review_item)
            }
            
        except Exception as e:
            logger.error("review_assignment_failed", {
                "review_id": review_id,
                "reviewer_id": reviewer_id,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def submit_review(self, review_id: str, review_result: ReviewResult) -> Dict[str, Any]:
        """
        Enviar resultado de revisión.
        
        Args:
            review_id: ID de revisión
            review_result: Resultado de la revisión
            
        Returns:
            Dict con confirmación
        """
        try:
            # Obtener item de revisión
            review_item = await self._get_review_item(review_id)
            if not review_item:
                return {
                    "success": False,
                    "error": "Review item not found"
                }
            
            # Actualizar estado
            review_item.status = ReviewStatus.COMPLETED
            review_item.completed_at = datetime.now(timezone.utc)
            review_item.review_notes = review_result.clinical_notes
            
            # Calcular tiempo de revisión
            if review_item.assigned_at:
                review_time = (review_item.completed_at - review_item.assigned_at).total_seconds() / 60
            else:
                review_time = (review_item.completed_at - review_item.created_at).total_seconds() / 60
            
            # Guardar resultado
            await self._save_review_item(review_item)
            await self._save_review_result(review_result)
            
            # Actualizar sesión si está disponible
            if self.session_manager:
                await self.session_manager.update_session_state(
                    review_item.session_id,
                    SessionState.COMPLETED,
                    additional_data={
                        "review_completed": True,
                        "reviewer_decision": review_result.decision,
                        "review_time_minutes": review_time
                    }
                )
            
            # Notificar completion
            await self._notify_review_completion(review_item, review_result)
            
            # Programar follow-up si es necesario
            if review_result.requires_follow_up:
                await self._schedule_follow_up(review_item, review_result)
            
            logger.audit("review_completed", {
                "review_id": review_id,
                "session_id": review_item.session_id,
                "reviewer_id": review_result.reviewer_id,
                "decision": review_result.decision,
                "review_time_minutes": review_time,
                "requires_follow_up": review_result.requires_follow_up
            })
            
            return {
                "success": True,
                "completed_at": review_item.completed_at.isoformat(),
                "review_time_minutes": review_time,
                "follow_up_scheduled": review_result.requires_follow_up
            }
            
        except Exception as e:
            logger.error("review_submission_failed", {
                "review_id": review_id,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_queue_status(self, role: Optional[ReviewerRole] = None) -> Dict[str, Any]:
        """
        Obtener estado actual de la cola.
        
        Args:
            role: Filtrar por rol específico
            
        Returns:
            Dict con estadísticas de cola
        """
        try:
            # Obtener todos los items pendientes
            pending_items = await self._get_pending_items(role)
            
            # Calcular estadísticas
            stats = {
                "total_pending": len(pending_items),
                "by_priority": {},
                "by_role": {},
                "overdue_items": 0,
                "average_wait_time": 0,
                "oldest_item_age": 0
            }
            
            now = datetime.now(timezone.utc)
            total_wait_time = 0
            
            for item in pending_items:
                # Por prioridad
                priority_name = item.priority.name
                if priority_name not in stats["by_priority"]:
                    stats["by_priority"][priority_name] = 0
                stats["by_priority"][priority_name] += 1
                
                # Por rol
                if item.required_role:
                    role_name = item.required_role.value
                    if role_name not in stats["by_role"]:
                        stats["by_role"][role_name] = 0
                    stats["by_role"][role_name] += 1
                
                # Calcular tiempos
                wait_time = (now - item.created_at).total_seconds() / 60
                total_wait_time += wait_time
                
                # Items vencidos
                max_wait = self.response_times[item.priority].total_seconds() / 60
                if wait_time > max_wait:
                    stats["overdue_items"] += 1
                
                # Item más antiguo
                if wait_time > stats["oldest_item_age"]:
                    stats["oldest_item_age"] = wait_time
            
            # Promedio de tiempo de espera
            if len(pending_items) > 0:
                stats["average_wait_time"] = total_wait_time / len(pending_items)
            
            return {
                "success": True,
                "timestamp": now.isoformat(),
                "statistics": stats,
                "health_status": self._assess_queue_health(stats)
            }
            
        except Exception as e:
            logger.error("queue_status_failed", {"error": str(e)})
            return {
                "success": False,
                "error": str(e)
            }
    
    def _initialize_medical_staff(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar personal médico disponible."""
        return {
            "dr_garcia": {
                "name": "Dr. García",
                "role": ReviewerRole.ATTENDING_PHYSICIAN,
                "specialties": ["internal_medicine", "emergency"],
                "availability": "24/7",
                "contact": "dr.garcia@hospital.com"
            },
            "nurse_martinez": {
                "name": "Enf. Martínez",
                "role": ReviewerRole.WOUND_CARE_SPECIALIST,
                "specialties": ["wound_care", "lpp_prevention"],
                "availability": "7am-7pm",
                "contact": "n.martinez@hospital.com"
            },
            "dr_rodriguez": {
                "name": "Dr. Rodríguez",
                "role": ReviewerRole.CLINICAL_SUPERVISOR,
                "specialties": ["quality_assurance", "protocols"],
                "availability": "8am-6pm",
                "contact": "dr.rodriguez@hospital.com"
            },
            "resident_lopez": {
                "name": "Dr. López (R3)",
                "role": ReviewerRole.MEDICAL_RESIDENT,
                "specialties": ["general_medicine"],
                "availability": "varies",
                "contact": "r.lopez@hospital.com"
            }
        }
    
    def _determine_review_priority(self, 
                                 triage_decision: TriageDecision,
                                 standardized_input: StandardizedInput) -> ReviewPriority:
        """Determinar prioridad de revisión."""
        # Emergencias médicas
        if "emergency" in triage_decision.flags:
            return ReviewPriority.EMERGENCY
        
        # Casos urgentes
        if "urgent" in triage_decision.flags or triage_decision.confidence < 0.5:
            return ReviewPriority.URGENT
        
        # Casos con riesgo de infección
        if "infection_risk" in triage_decision.flags:
            return ReviewPriority.HIGH
        
        # LPP grados altos
        if "lpp_grade_3" in triage_decision.flags or "lpp_grade_4" in triage_decision.flags:
            return ReviewPriority.HIGH
        
        # Casos con datos insuficientes
        if "missing_patient_code" in triage_decision.flags:
            return ReviewPriority.MEDIUM
        
        return ReviewPriority.LOW
    
    def _determine_required_role(self,
                               triage_decision: TriageDecision,
                               standardized_input: StandardizedInput) -> Optional[ReviewerRole]:
        """Determinar rol médico requerido."""
        # Emergencias requieren médico
        if "emergency" in triage_decision.flags:
            return ReviewerRole.ATTENDING_PHYSICIAN
        
        # LPP complejas requieren especialista
        if any(flag in triage_decision.flags for flag in ["lpp_grade_3", "lpp_grade_4", "infection_risk"]):
            return ReviewerRole.WOUND_CARE_SPECIALIST
        
        # Consultas de medicamentos requieren médico
        if "medication" in triage_decision.flags:
            return ReviewerRole.ATTENDING_PHYSICIAN
        
        # Casos generales pueden ser revisados por enfermero especialista
        return ReviewerRole.NURSE_SPECIALIST
    
    async def _create_review_item(self,
                                session_id: str,
                                priority: ReviewPriority,
                                required_role: Optional[ReviewerRole],
                                triage_decision: TriageDecision,
                                standardized_input: StandardizedInput) -> ReviewItem:
        """Crear item de revisión."""
        import uuid
        
        review_id = f"REV_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Crear contexto anonimizado
        context = {
            "triage_reason": triage_decision.reason,
            "triage_confidence": triage_decision.confidence,
            "triage_flags": triage_decision.flags,
            "input_type": standardized_input.input_type,
            "has_media": standardized_input.metadata.get("has_media", False),
            "source": standardized_input.metadata.get("source")
        }
        
        return ReviewItem(
            review_id=review_id,
            session_id=session_id,
            priority=priority,
            status=ReviewStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            required_role=required_role,
            context=context
        )
    
    async def _enqueue_for_review(self, review_item: ReviewItem) -> Dict[str, Any]:
        """Encolar item para revisión."""
        try:
            # Guardar item en Redis
            await self._save_review_item(review_item)
            
            # Añadir a cola por prioridad
            priority_queue = f"review_queue:priority_{review_item.priority.value}"
            await self.redis_client.lpush(priority_queue, review_item.review_id)
            
            # Añadir a cola por rol si es específico
            if review_item.required_role:
                role_queue = f"review_queue:role_{review_item.required_role.value}"
                await self.redis_client.lpush(role_queue, review_item.review_id)
            
            # Calcular posición en cola
            position = await self.redis_client.llen(priority_queue)
            
            return {
                "success": True,
                "position": position,
                "assigned_staff": None  # Se asignará automáticamente
            }
            
        except Exception as e:
            logger.error("enqueue_failed", {
                "review_id": review_item.review_id,
                "error": str(e)
            })
            raise
    
    async def _notify_medical_staff(self, review_item: ReviewItem):
        """Notificar al personal médico sobre nuevo caso."""
        if not self.slack_notifier:
            return
        
        try:
            # Determinar canal según prioridad
            if review_item.priority == ReviewPriority.EMERGENCY:
                channel = "emergency-medical"
            elif review_item.priority == ReviewPriority.URGENT:
                channel = "urgent-medical"
            else:
                channel = "medical-review"
            
            # Crear mensaje
            message = self._create_notification_message(review_item)
            
            # Enviar notificación
            await self.slack_notifier.send_message(channel, message)
            
            # Notificar específicamente al personal con el rol requerido
            if review_item.required_role:
                role_staff = [
                    staff for staff_id, staff in self.medical_staff.items()
                    if staff["role"] == review_item.required_role
                ]
                
                for staff in role_staff:
                    await self.slack_notifier.send_direct_message(
                        staff["contact"],
                        f"🏥 Nuevo caso requiere tu revisión (Prioridad: {review_item.priority.name})\n"
                        f"Review ID: {review_item.review_id}\n"
                        f"Usar comando: `/review assign {review_item.review_id}`"
                    )
            
        except Exception as e:
            logger.error("notification_failed", {
                "review_id": review_item.review_id,
                "error": str(e)
            })
    
    def _create_notification_message(self, review_item: ReviewItem) -> str:
        """Crear mensaje de notificación."""
        priority_emoji = {
            ReviewPriority.EMERGENCY: "🚨",
            ReviewPriority.URGENT: "⚡",
            ReviewPriority.HIGH: "🔺",
            ReviewPriority.MEDIUM: "🔸",
            ReviewPriority.LOW: "🔹"
        }
        
        emoji = priority_emoji.get(review_item.priority, "🔹")
        
        message = f"{emoji} **Nueva Revisión Médica Requerida**\n\n"
        message += f"**Review ID:** {review_item.review_id}\n"
        message += f"**Prioridad:** {review_item.priority.name}\n"
        message += f"**Rol requerido:** {review_item.required_role.value if review_item.required_role else 'Cualquiera'}\n"
        message += f"**Creado:** {review_item.created_at.strftime('%H:%M:%S')}\n"
        
        if review_item.context:
            message += f"**Contexto:** {review_item.context.get('triage_reason', 'N/A')}\n"
            message += f"**Confianza IA:** {review_item.context.get('triage_confidence', 0):.2%}\n"
        
        # Tiempo límite
        response_time = self.response_times.get(review_item.priority)
        if response_time:
            deadline = review_item.created_at + response_time
            message += f"**Responder antes de:** {deadline.strftime('%H:%M:%S')}\n"
        
        message += f"\n`/review assign {review_item.review_id}` para asignar"
        
        return message
    
    async def _save_review_item(self, review_item: ReviewItem):
        """Guardar item de revisión en Redis."""
        key = f"review_item:{review_item.review_id}"
        data = {
            "review_id": review_item.review_id,
            "session_id": review_item.session_id,
            "priority": review_item.priority.value,
            "status": review_item.status.value,
            "created_at": review_item.created_at.isoformat(),
            "assigned_to": review_item.assigned_to or "",
            "assigned_at": review_item.assigned_at.isoformat() if review_item.assigned_at else "",
            "completed_at": review_item.completed_at.isoformat() if review_item.completed_at else "",
            "review_notes": review_item.review_notes or "",
            "escalation_count": review_item.escalation_count,
            "last_escalated": review_item.last_escalated.isoformat() if review_item.last_escalated else "",
            "required_role": review_item.required_role.value if review_item.required_role else "",
            "context": json.dumps(review_item.context or {})
        }
        
        await self.redis_client.hset(key, mapping=data)
        
        # Set TTL para cleanup automático (7 días)
        await self.redis_client.expire(key, 604800)
    
    async def _get_review_item(self, review_id: str) -> Optional[ReviewItem]:
        """Obtener item de revisión desde Redis."""
        try:
            key = f"review_item:{review_id}"
            data = await self.redis_client.hgetall(key)
            
            if not data:
                return None
            
            return ReviewItem(
                review_id=data["review_id"],
                session_id=data["session_id"],
                priority=ReviewPriority(int(data["priority"])),
                status=ReviewStatus(data["status"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                assigned_to=data["assigned_to"] or None,
                assigned_at=datetime.fromisoformat(data["assigned_at"]) if data["assigned_at"] else None,
                completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
                review_notes=data["review_notes"] or None,
                escalation_count=int(data["escalation_count"]),
                last_escalated=datetime.fromisoformat(data["last_escalated"]) if data["last_escalated"] else None,
                required_role=ReviewerRole(data["required_role"]) if data["required_role"] else None,
                context=json.loads(data["context"]) if data["context"] else None
            )
            
        except Exception as e:
            logger.error("get_review_item_failed", {
                "review_id": review_id,
                "error": str(e)
            })
            return None
    
    async def _get_pending_items(self, role: Optional[ReviewerRole] = None) -> List[ReviewItem]:
        """Obtener items pendientes."""
        pending_items = []
        
        try:
            # Buscar en todas las colas de prioridad
            for priority in ReviewPriority:
                queue_key = f"review_queue:priority_{priority.value}"
                review_ids = await self.redis_client.lrange(queue_key, 0, -1)
                
                for review_id in review_ids:
                    review_id_str = review_id.decode() if isinstance(review_id, bytes) else review_id
                    item = await self._get_review_item(review_id_str)
                    
                    if item and item.status == ReviewStatus.PENDING:
                        # Filtrar por rol si se especifica
                        if role is None or item.required_role == role:
                            pending_items.append(item)
            
            return pending_items
            
        except Exception as e:
            logger.error("get_pending_items_failed", {"error": str(e)})
            return []
    
    async def _save_review_result(self, result: ReviewResult):
        """Guardar resultado de revisión."""
        key = f"review_result:{result.review_id}"
        data = {
            "review_id": result.review_id,
            "session_id": result.session_id,
            "reviewer_id": result.reviewer_id,
            "decision": result.decision,
            "clinical_notes": result.clinical_notes,
            "recommendations": json.dumps(result.recommendations),
            "requires_follow_up": str(result.requires_follow_up),
            "follow_up_timeline": result.follow_up_timeline or "",
            "reviewed_at": result.reviewed_at.isoformat(),
            "review_time_minutes": result.review_time_minutes
        }
        
        await self.redis_client.hset(key, mapping=data)
        await self.redis_client.expire(key, 2592000)  # 30 días
    
    def _estimate_wait_time(self, priority: ReviewPriority) -> str:
        """Estimar tiempo de espera."""
        response_time = self.response_times.get(priority)
        if response_time:
            minutes = int(response_time.total_seconds() / 60)
            if minutes < 60:
                return f"{minutes} minutos"
            else:
                hours = minutes // 60
                return f"{hours} horas"
        return "No estimado"
    
    def _calculate_next_escalation(self, review_item: ReviewItem) -> Optional[str]:
        """Calcular próxima escalación."""
        escalation_interval = self.escalation_intervals.get(review_item.priority)
        if escalation_interval:
            next_escalation = review_item.created_at + escalation_interval
            return next_escalation.isoformat()
        return None
    
    def _calculate_expected_completion(self, review_item: ReviewItem) -> str:
        """Calcular tiempo esperado de completion."""
        if review_item.assigned_at:
            base_time = review_item.assigned_at
        else:
            base_time = review_item.created_at
        
        response_time = self.response_times.get(review_item.priority, timedelta(hours=2))
        expected = base_time + response_time
        
        return expected.isoformat()
    
    def _assess_queue_health(self, stats: Dict[str, Any]) -> str:
        """Evaluar salud de la cola."""
        if stats["overdue_items"] == 0:
            return "healthy"
        elif stats["overdue_items"] <= stats["total_pending"] * 0.1:
            return "warning"
        else:
            return "critical"
    
    async def _notify_assignment(self, review_item: ReviewItem, reviewer: Dict[str, Any]):
        """Notificar asignación de revisor."""
        if self.slack_notifier:
            message = (
                f"✅ **Caso Asignado**\n"
                f"Review ID: {review_item.review_id}\n"
                f"Asignado a: {reviewer['name']}\n"
                f"Prioridad: {review_item.priority.name}"
            )
            
            channel = "medical-assignments"
            await self.slack_notifier.send_message(channel, message)
    
    async def _notify_review_completion(self, review_item: ReviewItem, review_result: ReviewResult):
        """Notificar completion de revisión."""
        if self.slack_notifier:
            message = (
                f"✅ **Revisión Completada**\n"
                f"Review ID: {review_item.review_id}\n"
                f"Decisión: {review_result.decision}\n"
                f"Tiempo de revisión: {review_result.review_time_minutes:.1f} min"
            )
            
            channel = "medical-completions"
            await self.slack_notifier.send_message(channel, message)
    
    async def _schedule_follow_up(self, review_item: ReviewItem, review_result: ReviewResult):
        """Programar seguimiento."""
        # En producción, esto integraría con sistema de scheduling
        logger.audit("follow_up_scheduled", {
            "review_id": review_item.review_id,
            "session_id": review_item.session_id,
            "timeline": review_result.follow_up_timeline,
            "reviewer": review_result.reviewer_id
        })
    
    async def _monitor_queue_timeouts(self):
        """Monitorear timeouts de cola."""
        while True:
            try:
                await asyncio.sleep(60)  # Revisar cada minuto
                
                pending_items = await self._get_pending_items()
                now = datetime.now(timezone.utc)
                
                for item in pending_items:
                    max_wait = self.response_times[item.priority]
                    wait_time = now - item.created_at
                    
                    if wait_time > max_wait:
                        await self._handle_timeout(item)
                        
            except Exception as e:
                logger.error("queue_timeout_monitor_failed", {"error": str(e)})
                await asyncio.sleep(300)  # Retry en 5 minutos
    
    async def _escalation_monitor(self):
        """Monitorear escalaciones automáticas."""
        while True:
            try:
                await asyncio.sleep(120)  # Revisar cada 2 minutos
                
                pending_items = await self._get_pending_items()
                now = datetime.now(timezone.utc)
                
                for item in pending_items:
                    escalation_interval = self.escalation_intervals[item.priority]
                    
                    # Tiempo desde última escalación o creación
                    last_action = item.last_escalated or item.created_at
                    time_since_last = now - last_action
                    
                    if time_since_last > escalation_interval:
                        await self._escalate_review(item)
                        
            except Exception as e:
                logger.error("escalation_monitor_failed", {"error": str(e)})
                await asyncio.sleep(300)
    
    async def _handle_timeout(self, review_item: ReviewItem):
        """Manejar timeout de revisión."""
        logger.warning("review_timeout", {
            "review_id": review_item.review_id,
            "priority": review_item.priority.name,
            "created_at": review_item.created_at.isoformat()
        })
        
        # Escalar automáticamente
        await self._escalate_review(review_item)
    
    async def _escalate_review(self, review_item: ReviewItem):
        """Escalar revisión a nivel superior."""
        try:
            review_item.escalation_count += 1
            review_item.last_escalated = datetime.now(timezone.utc)
            
            # Determinar nuevo rol requerido (escalar)
            escalation_map = {
                ReviewerRole.MEDICAL_RESIDENT: ReviewerRole.ATTENDING_PHYSICIAN,
                ReviewerRole.NURSE_SPECIALIST: ReviewerRole.WOUND_CARE_SPECIALIST,
                ReviewerRole.WOUND_CARE_SPECIALIST: ReviewerRole.ATTENDING_PHYSICIAN,
                ReviewerRole.ATTENDING_PHYSICIAN: ReviewerRole.CLINICAL_SUPERVISOR
            }
            
            if review_item.required_role in escalation_map:
                review_item.required_role = escalation_map[review_item.required_role]
            
            # Guardar cambios
            await self._save_review_item(review_item)
            
            # Notificar escalación
            if self.slack_notifier:
                message = (
                    f"⚠️ **ESCALACIÓN AUTOMÁTICA**\n"
                    f"Review ID: {review_item.review_id}\n"
                    f"Escalación #{review_item.escalation_count}\n"
                    f"Nuevo rol requerido: {review_item.required_role.value if review_item.required_role else 'N/A'}\n"
                    f"Prioridad: {review_item.priority.name}"
                )
                
                await self.slack_notifier.send_message("medical-escalations", message)
            
            logger.audit("review_escalated", {
                "review_id": review_item.review_id,
                "escalation_count": review_item.escalation_count,
                "new_required_role": review_item.required_role.value if review_item.required_role else None
            })
            
        except Exception as e:
            logger.error("escalation_failed", {
                "review_id": review_item.review_id,
                "error": str(e)
            })


class HumanReviewQueueFactory:
    """Factory para crear instancias del sistema de revisión humana."""
    
    @staticmethod
    async def create_queue(config: Optional[Dict[str, Any]] = None) -> HumanReviewQueue:
        """Crear sistema de revisión humana."""
        config = config or {}
        
        # Crear componentes opcionales
        slack_notifier = SlackNotifier() if config.get("use_slack", True) else None
        session_manager = SessionManager() if config.get("use_session_manager", True) else None
        
        queue = HumanReviewQueue(
            redis_url=config.get("redis_url"),
            slack_notifier=slack_notifier,
            session_manager=session_manager
        )
        
        await queue.initialize()
        return queue