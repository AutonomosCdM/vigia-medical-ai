"""
Input Queue - Capa 1: Isolación Temporal con Encryption
Buffer temporal entre entrada y procesamiento con session tokens.

Características:
- Buffer temporal entre entrada y procesamiento
- Session tokens para aislamiento
- Timeout automático (15 minutos)
- Encryption at rest
- No logging de contenido PII
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from cryptography.fernet import Fernet
import redis.asyncio as redis

from .input_packager import StandardizedInput
from ..utils.secure_logger import SecureLogger

logger = SecureLogger("input_queue")


class QueueStatus(Enum):
    """Estados de procesamiento en la queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class QueueItem:
    """Item en la Input Queue."""
    session_id: str
    encrypted_payload: str
    created_at: datetime
    expires_at: datetime
    status: QueueStatus
    retry_count: int = 0
    last_accessed: Optional[datetime] = None
    processing_node: Optional[str] = None


class InputQueue:
    """
    Input Queue con encryption y session management.
    Proporciona buffer temporal seguro entre entrada y procesamiento.
    """
    
    def __init__(self, redis_url: Optional[str] = None, encryption_key: Optional[str] = None):
        """
        Inicializar Input Queue.
        
        Args:
            redis_url: URL de Redis para persistencia
            encryption_key: Clave de encriptación (se genera si no se proporciona)
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Configurar encryption
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Generar clave temporal (en producción debe ser persistente)
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            logger.warning("using_temporary_encryption_key", {
                "component": "input_queue",
                "recommendation": "use_persistent_key_in_production"
            })
        
        # Configuración de timeouts
        self.default_timeout = timedelta(minutes=15)
        self.max_retry_count = 3
        self.cleanup_interval = timedelta(minutes=5)
        
        # Redis client será inicializado asincrónicamente
        self.redis_client = None
        
        logger.audit("input_queue_initialized", {
            "component": "layer1_input_queue",
            "timeout_minutes": 15,
            "max_retries": self.max_retry_count,
            "encryption_enabled": True
        })
    
    async def initialize(self):
        """Inicializar conexión Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            logger.audit("input_queue_redis_connected", {
                "redis_url": self.redis_url.split('@')[-1]  # Log sin credenciales
            })
            
            # Iniciar tarea de cleanup
            asyncio.create_task(self._cleanup_expired_items())
            
        except Exception as e:
            logger.error("input_queue_redis_connection_failed", {
                "error": str(e),
                "redis_url": self.redis_url.split('@')[-1]
            })
            raise
    
    async def enqueue(self, standardized_input: StandardizedInput) -> Dict[str, Any]:
        """
        Añadir input estandarizado a la queue con encryption.
        
        Args:
            standardized_input: Input estandarizado del Input Packager
            
        Returns:
            Dict con información de enqueue
        """
        try:
            if not self.redis_client:
                await self.initialize()
            
            session_id = standardized_input.session_id
            
            # Encriptar payload
            payload_json = standardized_input.to_json()
            encrypted_payload = self._encrypt_payload(payload_json)
            
            # Crear queue item
            now = datetime.now(timezone.utc)
            expires_at = now + self.default_timeout
            
            queue_item = QueueItem(
                session_id=session_id,
                encrypted_payload=encrypted_payload,
                created_at=now,
                expires_at=expires_at,
                status=QueueStatus.PENDING
            )
            
            # Guardar en Redis
            queue_key = f"input_queue:{session_id}"
            queue_data = {
                "session_id": session_id,
                "encrypted_payload": encrypted_payload,
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "status": QueueStatus.PENDING.value,
                "retry_count": 0
            }
            
            # Usar pipeline para operaciones atómicas
            pipe = self.redis_client.pipeline()
            await pipe.hset(queue_key, mapping=queue_data)
            await pipe.expire(queue_key, int(self.default_timeout.total_seconds()))
            await pipe.sadd("input_queue:sessions", session_id)
            await pipe.execute()
            
            # Log de enqueue exitoso (sin datos PII)
            logger.audit("input_enqueued", {
                "session_id": session_id,
                "input_type": standardized_input.input_type,
                "expires_at": expires_at.isoformat(),
                "queue_position": await self._get_queue_position(session_id)
            })
            
            return {
                "success": True,
                "session_id": session_id,
                "expires_at": expires_at.isoformat(),
                "queue_key": queue_key,
                "estimated_processing_time": "30_seconds"
            }
            
        except Exception as e:
            logger.error("input_enqueue_failed", {
                "session_id": standardized_input.session_id,
                "error": str(e)
            })
            return {
                "success": False,
                "error": str(e),
                "session_id": standardized_input.session_id
            }
    
    async def dequeue(self, session_id: str, processor_id: str) -> Optional[StandardizedInput]:
        """
        Obtener y marcar como procesando un input de la queue.
        
        Args:
            session_id: ID de sesión a procesar
            processor_id: ID del procesador que toma el item
            
        Returns:
            StandardizedInput decriptado o None si no disponible
        """
        try:
            if not self.redis_client:
                await self.initialize()
            
            queue_key = f"input_queue:{session_id}"
            
            # Obtener item de Redis
            queue_data = await self.redis_client.hgetall(queue_key)
            if not queue_data:
                return None
            
            # Verificar si no ha expirado
            expires_at = datetime.fromisoformat(queue_data["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                await self._mark_expired(session_id)
                return None
            
            # Verificar estado
            current_status = QueueStatus(queue_data["status"])
            if current_status != QueueStatus.PENDING:
                logger.warning("dequeue_invalid_status", {
                    "session_id": session_id,
                    "current_status": current_status.value,
                    "processor_id": processor_id
                })
                return None
            
            # Marcar como procesando
            now = datetime.now(timezone.utc)
            await self.redis_client.hset(queue_key, mapping={
                "status": QueueStatus.PROCESSING.value,
                "last_accessed": now.isoformat(),
                "processing_node": processor_id
            })
            
            # Decriptar payload
            encrypted_payload = queue_data["encrypted_payload"]
            decrypted_json = self._decrypt_payload(encrypted_payload)
            
            # Reconstruir StandardizedInput
            payload_dict = json.loads(decrypted_json)
            standardized_input = StandardizedInput(**payload_dict)
            
            logger.audit("input_dequeued", {
                "session_id": session_id,
                "processor_id": processor_id,
                "input_type": standardized_input.input_type,
                "processing_started": now.isoformat()
            })
            
            return standardized_input
            
        except Exception as e:
            logger.error("input_dequeue_failed", {
                "session_id": session_id,
                "processor_id": processor_id,
                "error": str(e)
            })
            return None
    
    async def mark_completed(self, session_id: str, result: Dict[str, Any]) -> bool:
        """
        Marcar procesamiento como completado.
        
        Args:
            session_id: ID de sesión completada
            result: Resultado del procesamiento
            
        Returns:
            True si marcado exitosamente
        """
        try:
            queue_key = f"input_queue:{session_id}"
            now = datetime.now(timezone.utc)
            
            # Actualizar estado
            await self.redis_client.hset(queue_key, mapping={
                "status": QueueStatus.COMPLETED.value,
                "completed_at": now.isoformat(),
                "result_summary": json.dumps({
                    "success": result.get("success", False),
                    "processing_time": result.get("processing_time", 0),
                    "next_stage": result.get("next_stage", "unknown")
                })
            })
            
            # Remover de sessions activas
            await self.redis_client.srem("input_queue:sessions", session_id)
            
            # Programar cleanup en 1 hora
            await self.redis_client.expire(queue_key, 3600)
            
            logger.audit("input_completed", {
                "session_id": session_id,
                "completed_at": now.isoformat(),
                "success": result.get("success", False)
            })
            
            return True
            
        except Exception as e:
            logger.error("mark_completed_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    async def mark_failed(self, session_id: str, error: str, retry: bool = True) -> bool:
        """
        Marcar procesamiento como fallido.
        
        Args:
            session_id: ID de sesión fallida
            error: Descripción del error
            retry: Si debe intentar retry
            
        Returns:
            True si marcado exitosamente
        """
        try:
            queue_key = f"input_queue:{session_id}"
            now = datetime.now(timezone.utc)
            
            # Obtener estado actual
            queue_data = await self.redis_client.hgetall(queue_key)
            retry_count = int(queue_data.get("retry_count", 0))
            
            # Decidir si hacer retry
            should_retry = retry and retry_count < self.max_retry_count
            new_status = QueueStatus.PENDING if should_retry else QueueStatus.FAILED
            new_retry_count = retry_count + 1 if should_retry else retry_count
            
            # Actualizar estado
            update_data = {
                "status": new_status.value,
                "retry_count": new_retry_count,
                "last_error": error,
                "last_failed_at": now.isoformat()
            }
            
            if not should_retry:
                update_data["failed_at"] = now.isoformat()
                # Remover de sessions activas
                await self.redis_client.srem("input_queue:sessions", session_id)
            
            await self.redis_client.hset(queue_key, mapping=update_data)
            
            logger.audit("input_failed", {
                "session_id": session_id,
                "error": error,
                "retry_count": new_retry_count,
                "will_retry": should_retry,
                "final_status": new_status.value
            })
            
            return True
            
        except Exception as e:
            logger.error("mark_failed_error", {
                "session_id": session_id,
                "error": str(e)
            })
            return False
    
    async def get_queue_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener estado actual de un item en la queue.
        
        Args:
            session_id: ID de sesión
            
        Returns:
            Dict con estado o None si no existe
        """
        try:
            queue_key = f"input_queue:{session_id}"
            queue_data = await self.redis_client.hgetall(queue_key)
            
            if not queue_data:
                return None
            
            return {
                "session_id": session_id,
                "status": queue_data.get("status"),
                "created_at": queue_data.get("created_at"),
                "expires_at": queue_data.get("expires_at"),
                "retry_count": int(queue_data.get("retry_count", 0)),
                "last_accessed": queue_data.get("last_accessed"),
                "processing_node": queue_data.get("processing_node")
            }
            
        except Exception as e:
            logger.error("get_queue_status_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            return None
    
    async def list_pending_sessions(self) -> List[str]:
        """Listar todas las sesiones pendientes."""
        try:
            sessions = await self.redis_client.smembers("input_queue:sessions")
            return [session.decode() if isinstance(session, bytes) else session for session in sessions]
        except Exception as e:
            logger.error("list_pending_sessions_failed", {"error": str(e)})
            return []
    
    def _encrypt_payload(self, payload_json: str) -> str:
        """Encriptar payload JSON."""
        return self.fernet.encrypt(payload_json.encode()).decode()
    
    def _decrypt_payload(self, encrypted_payload: str) -> str:
        """Decriptar payload."""
        return self.fernet.decrypt(encrypted_payload.encode()).decode()
    
    async def _get_queue_position(self, session_id: str) -> int:
        """Obtener posición aproximada en la queue."""
        try:
            sessions = await self.list_pending_sessions()
            if session_id in sessions:
                return sessions.index(session_id) + 1
            return 0
        except:
            return 0
    
    async def _mark_expired(self, session_id: str):
        """Marcar item como expirado."""
        try:
            queue_key = f"input_queue:{session_id}"
            now = datetime.now(timezone.utc)
            
            await self.redis_client.hset(queue_key, mapping={
                "status": QueueStatus.EXPIRED.value,
                "expired_at": now.isoformat()
            })
            
            await self.redis_client.srem("input_queue:sessions", session_id)
            
            logger.audit("input_expired", {
                "session_id": session_id,
                "expired_at": now.isoformat()
            })
            
        except Exception as e:
            logger.error("mark_expired_failed", {
                "session_id": session_id,
                "error": str(e)
            })
    
    async def _cleanup_expired_items(self):
        """Tarea de limpieza de items expirados."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                
                sessions = await self.list_pending_sessions()
                now = datetime.now(timezone.utc)
                expired_count = 0
                
                for session_id in sessions:
                    status = await self.get_queue_status(session_id)
                    if status:
                        expires_at = datetime.fromisoformat(status["expires_at"])
                        if now > expires_at:
                            await self._mark_expired(session_id)
                            expired_count += 1
                
                if expired_count > 0:
                    logger.audit("cleanup_expired_items", {
                        "expired_count": expired_count,
                        "cleanup_time": now.isoformat()
                    })
                    
            except Exception as e:
                logger.error("cleanup_task_failed", {"error": str(e)})
                await asyncio.sleep(60)  # Retry en 1 minuto


class InputQueueManager:
    """Manager para operaciones de Input Queue."""
    
    def __init__(self, redis_url: Optional[str] = None, encryption_key: Optional[str] = None):
        """Inicializar manager."""
        self.queue = InputQueue(redis_url, encryption_key)
    
    async def initialize(self):
        """Inicializar queue."""
        await self.queue.initialize()
    
    async def process_standardized_input(self, standardized_input: StandardizedInput) -> Dict[str, Any]:
        """
        Procesar input estandarizado a través de la queue.
        
        Args:
            standardized_input: Input estandarizado
            
        Returns:
            Dict con resultado de enqueue
        """
        return await self.queue.enqueue(standardized_input)
    
    async def get_next_for_processing(self, processor_id: str) -> Optional[StandardizedInput]:
        """
        Obtener próximo input para procesamiento.
        
        Args:
            processor_id: ID del procesador
            
        Returns:
            StandardizedInput o None si no hay items pendientes
        """
        sessions = await self.queue.list_pending_sessions()
        
        for session_id in sessions:
            status = await self.queue.get_queue_status(session_id)
            if status and status["status"] == QueueStatus.PENDING.value:
                input_item = await self.queue.dequeue(session_id, processor_id)
                if input_item:
                    return input_item
        
        return None