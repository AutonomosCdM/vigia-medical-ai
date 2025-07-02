"""
Session Manager - Aislamiento Temporal por Sesión
Gestiona el ciclo de vida completo de sesiones de procesamiento médico.

Características:
- Aislamiento temporal estricto (15 minutos max)
- Tracking de estado por sesión
- Cleanup automático de datos temporales
- Session tokens únicos
- Audit trail completo
"""

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import redis.asyncio as redis

from ..utils.secure_logger import SecureLogger

logger = SecureLogger("session_manager")


class SessionState(Enum):
    """Estados de sesión en el pipeline."""
    CREATED = "created"
    INPUT_RECEIVED = "input_received"
    QUEUED = "queued"
    TRIAGING = "triaging"
    PROCESSING = "processing"
    REVIEW_PENDING = "review_pending"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CLEANUP = "cleanup"


class SessionType(Enum):
    """Tipos de sesión según ruta de procesamiento."""
    CLINICAL_IMAGE = "clinical_image"
    MEDICAL_QUERY = "medical_query"
    HUMAN_ESCALATION = "human_escalation"
    EMERGENCY = "emergency"


@dataclass
class SessionMetadata:
    """Metadata de sesión para tracking."""
    session_id: str
    session_type: SessionType
    state: SessionState
    created_at: datetime
    expires_at: datetime
    source: str
    input_type: str
    processing_layer: Optional[str] = None
    assigned_processor: Optional[str] = None
    escalation_reason: Optional[str] = None
    audit_events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.audit_events is None:
            self.audit_events = []


class SessionManager:
    """
    Gestor de sesiones con aislamiento temporal.
    Controla el ciclo de vida completo de procesamiento médico.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Inicializar Session Manager.
        
        Args:
            redis_url: URL de Redis para persistencia de sesiones
        """
        self.redis_url = redis_url or "redis://localhost:6379/1"  # DB separada para sesiones
        self.redis_client = None
        
        # Configuración de timeouts
        self.default_session_timeout = timedelta(minutes=15)
        self.emergency_session_timeout = timedelta(minutes=30)
        self.cleanup_interval = timedelta(minutes=2)
        
        # Límites de sesiones concurrentes por tipo
        self.max_concurrent_sessions = {
            SessionType.CLINICAL_IMAGE: 50,
            SessionType.MEDICAL_QUERY: 100,
            SessionType.HUMAN_ESCALATION: 20,
            SessionType.EMERGENCY: 10
        }
        
        logger.audit("session_manager_initialized", {
            "component": "session_manager",
            "default_timeout_minutes": 15,
            "emergency_timeout_minutes": 30,
            "max_concurrent_clinical": 50
        })
    
    async def initialize(self):
        """Inicializar conexión Redis y tareas de mantenimiento."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            logger.audit("session_manager_redis_connected", {
                "redis_url": self.redis_url.split('@')[-1]
            })
            
            # Iniciar tareas de mantenimiento
            asyncio.create_task(self._cleanup_expired_sessions())
            asyncio.create_task(self._monitor_session_health())
            
        except Exception as e:
            logger.error("session_manager_redis_failed", {
                "error": str(e),
                "redis_url": self.redis_url.split('@')[-1]
            })
            raise
    
    async def create_session(self, 
                           input_data: Dict[str, Any], 
                           session_type: SessionType,
                           emergency: bool = False) -> Dict[str, Any]:
        """
        Crear nueva sesión de procesamiento.
        
        Args:
            input_data: Datos de entrada (ya anonimizados)
            session_type: Tipo de sesión
            emergency: Si es emergencia médica
            
        Returns:
            Dict con información de sesión creada
        """
        try:
            if not self.redis_client:
                await self.initialize()
            
            # Verificar límites de concurrencia
            current_count = await self._get_active_session_count(session_type)
            max_allowed = self.max_concurrent_sessions[session_type]
            
            if current_count >= max_allowed:
                return {
                    "success": False,
                    "error": "max_concurrent_sessions_reached",
                    "current_count": current_count,
                    "max_allowed": max_allowed
                }
            
            # Generar session ID
            session_id = self._generate_session_id(session_type, emergency)
            
            # Configurar timeout
            timeout = self.emergency_session_timeout if emergency else self.default_session_timeout
            
            # Crear metadata
            now = datetime.now(timezone.utc)
            metadata = SessionMetadata(
                session_id=session_id,
                session_type=session_type,
                state=SessionState.CREATED,
                created_at=now,
                expires_at=now + timeout,
                source=input_data.get('source', 'unknown'),
                input_type=input_data.get('input_type', 'unknown')
            )
            
            # Registrar evento inicial
            await self._add_audit_event(metadata, "session_created", {
                "session_type": session_type.value,
                "emergency": emergency,
                "expires_at": metadata.expires_at.isoformat()
            })
            
            # Guardar en Redis
            session_key = f"session:{session_id}"
            await self.redis_client.hset(session_key, mapping={
                "metadata": json.dumps(asdict(metadata), default=str),
                "created_at": now.isoformat(),
                "expires_at": metadata.expires_at.isoformat(),
                "state": SessionState.CREATED.value,
                "type": session_type.value,
                "emergency": str(emergency)
            })
            
            # Set expiration
            await self.redis_client.expire(session_key, int(timeout.total_seconds()))
            
            # Añadir a set de sesiones activas
            await self.redis_client.sadd(f"active_sessions:{session_type.value}", session_id)
            
            logger.audit("session_created", {
                "session_id": session_id,
                "session_type": session_type.value,
                "emergency": emergency,
                "expires_at": metadata.expires_at.isoformat(),
                "active_count": current_count + 1
            })
            
            return {
                "success": True,
                "session_id": session_id,
                "expires_at": metadata.expires_at.isoformat(),
                "timeout_minutes": int(timeout.total_seconds() / 60),
                "session_type": session_type.value
            }
            
        except Exception as e:
            logger.error("session_creation_failed", {
                "session_type": session_type.value,
                "error": str(e)
            })
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_session_state(self, 
                                 session_id: str, 
                                 new_state: SessionState,
                                 processor_id: Optional[str] = None,
                                 additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Actualizar estado de sesión.
        
        Args:
            session_id: ID de sesión
            new_state: Nuevo estado
            processor_id: ID del procesador (opcional)
            additional_data: Datos adicionales (opcional)
            
        Returns:
            True si actualización exitosa
        """
        try:
            session_key = f"session:{session_id}"
            
            # Verificar que la sesión existe
            session_data = await self.redis_client.hgetall(session_key)
            if not session_data:
                logger.warning("session_not_found", {"session_id": session_id})
                return False
            
            # Verificar que no ha expirado
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                await self._mark_session_expired(session_id)
                return False
            
            # Actualizar estado
            now = datetime.now(timezone.utc)
            update_data = {
                "state": new_state.value,
                "last_updated": now.isoformat()
            }
            
            if processor_id:
                update_data["assigned_processor"] = processor_id
            
            if new_state == SessionState.PROCESSING:
                update_data["processing_started"] = now.isoformat()
            elif new_state in [SessionState.COMPLETED, SessionState.FAILED]:
                update_data["finished_at"] = now.isoformat()
            
            await self.redis_client.hset(session_key, mapping=update_data)
            
            # Actualizar metadata con audit event
            metadata = await self._get_session_metadata(session_id)
            if metadata:
                await self._add_audit_event(metadata, "state_changed", {
                    "old_state": metadata.state.value,
                    "new_state": new_state.value,
                    "processor_id": processor_id,
                    "additional_data": additional_data
                })
                
                metadata.state = new_state
                if processor_id:
                    metadata.assigned_processor = processor_id
                
                await self.redis_client.hset(session_key, "metadata", 
                                           json.dumps(asdict(metadata), default=str))
            
            logger.audit("session_state_updated", {
                "session_id": session_id,
                "new_state": new_state.value,
                "processor_id": processor_id,
                "updated_at": now.isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error("session_state_update_failed", {
                "session_id": session_id,
                "new_state": new_state.value,
                "error": str(e)
            })
            return False
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información completa de sesión.
        
        Args:
            session_id: ID de sesión
            
        Returns:
            Dict con información de sesión o None si no existe
        """
        try:
            session_key = f"session:{session_id}"
            session_data = await self.redis_client.hgetall(session_key)
            
            if not session_data:
                return None
            
            # Verificar expiración
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            is_expired = datetime.now(timezone.utc) > expires_at
            
            return {
                "session_id": session_id,
                "state": session_data.get("state"),
                "type": session_data.get("type"),
                "emergency": session_data.get("emergency") == "True",
                "created_at": session_data.get("created_at"),
                "expires_at": session_data.get("expires_at"),
                "last_updated": session_data.get("last_updated"),
                "assigned_processor": session_data.get("assigned_processor"),
                "processing_started": session_data.get("processing_started"),
                "finished_at": session_data.get("finished_at"),
                "is_expired": is_expired,
                "time_remaining": max(0, (expires_at - datetime.now(timezone.utc)).total_seconds()) if not is_expired else 0
            }
            
        except Exception as e:
            logger.error("get_session_info_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return None
    
    async def extend_session(self, session_id: str, additional_minutes: int = 5) -> bool:
        """
        Extender tiempo de vida de sesión.
        
        Args:
            session_id: ID de sesión
            additional_minutes: Minutos adicionales
            
        Returns:
            True si extensión exitosa
        """
        try:
            session_key = f"session:{session_id}"
            session_data = await self.redis_client.hgetall(session_key)
            
            if not session_data:
                return False
            
            # Calcular nueva expiración
            current_expires = datetime.fromisoformat(session_data["expires_at"])
            new_expires = current_expires + timedelta(minutes=additional_minutes)
            
            # Actualizar
            await self.redis_client.hset(session_key, mapping={
                "expires_at": new_expires.isoformat(),
                "extended_at": datetime.now(timezone.utc).isoformat()
            })
            
            # Actualizar TTL en Redis
            remaining_seconds = (new_expires - datetime.now(timezone.utc)).total_seconds()
            await self.redis_client.expire(session_key, int(remaining_seconds))
            
            logger.audit("session_extended", {
                "session_id": session_id,
                "additional_minutes": additional_minutes,
                "new_expires": new_expires.isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error("session_extension_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    async def cleanup_session(self, session_id: str, reason: str = "normal_completion") -> bool:
        """
        Limpiar sesión y datos temporales.
        
        Args:
            session_id: ID de sesión
            reason: Razón de cleanup
            
        Returns:
            True si cleanup exitoso
        """
        try:
            session_key = f"session:{session_id}"
            
            # Obtener metadata antes de limpiar
            metadata = await self._get_session_metadata(session_id)
            if metadata:
                await self._add_audit_event(metadata, "session_cleanup", {
                    "reason": reason,
                    "final_state": metadata.state.value
                })
            
            # Remover de sets activos
            for session_type in SessionType:
                await self.redis_client.srem(f"active_sessions:{session_type.value}", session_id)
            
            # Marcar como cleanup pero mantener por audit
            await self.redis_client.hset(session_key, mapping={
                "state": SessionState.CLEANUP.value,
                "cleanup_at": datetime.now(timezone.utc).isoformat(),
                "cleanup_reason": reason
            })
            
            # Programar eliminación final en 24 horas
            await self.redis_client.expire(session_key, 86400)
            
            logger.audit("session_cleaned_up", {
                "session_id": session_id,
                "reason": reason
            })
            
            return True
            
        except Exception as e:
            logger.error("session_cleanup_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    async def list_active_sessions(self, session_type: Optional[SessionType] = None) -> List[str]:
        """
        Listar sesiones activas.
        
        Args:
            session_type: Filtrar por tipo (opcional)
            
        Returns:
            Lista de session IDs activos
        """
        try:
            if session_type:
                sessions = await self.redis_client.smembers(f"active_sessions:{session_type.value}")
                return [s.decode() if isinstance(s, bytes) else s for s in sessions]
            else:
                all_sessions = []
                for stype in SessionType:
                    sessions = await self.redis_client.smembers(f"active_sessions:{stype.value}")
                    all_sessions.extend([s.decode() if isinstance(s, bytes) else s for s in sessions])
                return all_sessions
                
        except Exception as e:
            logger.error("list_active_sessions_failed", {"error": str(e)})
            return []
    
    def _generate_session_id(self, session_type: SessionType, emergency: bool = False) -> str:
        """Generar session ID único."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        prefix = "EMR" if emergency else "SES"
        type_code = session_type.value[:3].upper()
        
        return f"VIGIA_{prefix}_{type_code}_{timestamp}_{unique_id}"
    
    async def _get_active_session_count(self, session_type: SessionType) -> int:
        """Obtener número de sesiones activas del tipo."""
        try:
            return await self.redis_client.scard(f"active_sessions:{session_type.value}")
        except:
            return 0
    
    async def _get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Obtener metadata de sesión."""
        try:
            session_key = f"session:{session_id}"
            metadata_json = await self.redis_client.hget(session_key, "metadata")
            
            if metadata_json:
                data = json.loads(metadata_json)
                # Reconstruir datetime objects
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])
                data['state'] = SessionState(data['state'])
                data['session_type'] = SessionType(data['session_type'])
                
                return SessionMetadata(**data)
            return None
            
        except Exception as e:
            logger.error("get_session_metadata_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return None
    
    async def _add_audit_event(self, metadata: SessionMetadata, event_type: str, event_data: Dict[str, Any]):
        """Añadir evento de audit a sesión."""
        try:
            audit_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "session_id": metadata.session_id,
                "data": event_data
            }
            
            metadata.audit_events.append(audit_event)
            
            # Log del evento
            logger.audit(f"session_{event_type}", {
                "session_id": metadata.session_id,
                **event_data
            })
            
        except Exception as e:
            logger.error("add_audit_event_failed", {
                "session_id": metadata.session_id,
                "event_type": event_type,
                "error": str(e)
            })
    
    async def _mark_session_expired(self, session_id: str):
        """Marcar sesión como expirada."""
        try:
            session_key = f"session:{session_id}"
            now = datetime.now(timezone.utc)
            
            await self.redis_client.hset(session_key, mapping={
                "state": SessionState.EXPIRED.value,
                "expired_at": now.isoformat()
            })
            
            # Remover de sesiones activas
            for session_type in SessionType:
                await self.redis_client.srem(f"active_sessions:{session_type.value}", session_id)
            
            logger.audit("session_expired", {
                "session_id": session_id,
                "expired_at": now.isoformat()
            })
            
        except Exception as e:
            logger.error("mark_session_expired_failed", {
                "session_id": session_id,
                "error": str(e)
            })
    
    async def _cleanup_expired_sessions(self):
        """Tarea de limpieza de sesiones expiradas."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                
                expired_count = 0
                now = datetime.now(timezone.utc)
                
                # Revisar todas las sesiones activas
                for session_type in SessionType:
                    sessions = await self.list_active_sessions(session_type)
                    
                    for session_id in sessions:
                        info = await self.get_session_info(session_id)
                        if info and info["is_expired"]:
                            await self._mark_session_expired(session_id)
                            expired_count += 1
                
                if expired_count > 0:
                    logger.audit("cleanup_expired_sessions", {
                        "expired_count": expired_count,
                        "cleanup_time": now.isoformat()
                    })
                    
            except Exception as e:
                logger.error("cleanup_task_failed", {"error": str(e)})
                await asyncio.sleep(60)
    
    async def _monitor_session_health(self):
        """Monitorear salud del sistema de sesiones."""
        while True:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                health_data = {}
                total_active = 0
                
                for session_type in SessionType:
                    count = await self._get_active_session_count(session_type)
                    health_data[session_type.value] = {
                        "active_count": count,
                        "max_allowed": self.max_concurrent_sessions[session_type],
                        "utilization": count / self.max_concurrent_sessions[session_type]
                    }
                    total_active += count
                
                logger.audit("session_health_check", {
                    "total_active_sessions": total_active,
                    "session_types": health_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error("health_monitor_failed", {"error": str(e)})
                await asyncio.sleep(60)