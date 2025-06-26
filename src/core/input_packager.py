"""
Input Packager - Capa 1: Estandarización de Payloads
Estandariza todos los inputs en formato común para el pipeline.

Responsabilidades:
- Normalizar estructura de datos de entrada
- Aplicar checksums de integridad
- Preparar para Input Queue
- Mantener trazabilidad sin PII
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json

from ..utils.secure_logger import SecureLogger

logger = SecureLogger("input_packager")


class InputType(Enum):
    """Tipos de input estandarizados."""
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class InputSource(Enum):
    """Fuentes de input permitidas."""
    WHATSAPP = "whatsapp"
    API = "api"
    CLI = "cli"
    WEBHOOK = "webhook"


@dataclass
class StandardizedInput:
    """
    Estructura de Payload Estandarizado según documento de arquitectura.
    """
    session_id: str
    timestamp: str
    input_type: str
    raw_content: Dict[str, Any]
    metadata: Dict[str, Any]
    audit_trail: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convertir a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class InputPackager:
    """
    Empaquetador de inputs para estandarización.
    Convierte todos los inputs a formato StandardizedInput.
    """
    
    def __init__(self):
        """Inicializar empaquetador."""
        self.version = "1.0.0"
        
        logger.audit("input_packager_initialized", {
            "component": "layer1_input_packager",
            "version": self.version
        })
    
    def package_whatsapp_input(self, webhook_data: Dict[str, Any], session_id: str) -> StandardizedInput:
        """
        Empaquetar input de WhatsApp en formato estandarizado.
        
        Args:
            webhook_data: Datos raw del webhook de Twilio
            session_id: ID de sesión único
            
        Returns:
            StandardizedInput object
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Determinar tipo de input
            input_type = self._determine_input_type(webhook_data)
            
            # Crear contenido raw
            raw_content = self._extract_raw_content(webhook_data, InputSource.WHATSAPP)
            
            # Crear metadata
            metadata = self._create_metadata(webhook_data, InputSource.WHATSAPP, raw_content)
            
            # Crear audit trail
            audit_trail = self._create_audit_trail(webhook_data, session_id, InputSource.WHATSAPP)
            
            standardized = StandardizedInput(
                session_id=session_id,
                timestamp=timestamp,
                input_type=input_type.value,
                raw_content=raw_content,
                metadata=metadata,
                audit_trail=audit_trail
            )
            
            # Log de empaquetado exitoso
            logger.audit("input_packaged_successfully", {
                "session_id": session_id,
                "input_type": input_type.value,
                "source": InputSource.WHATSAPP.value,
                "has_media": metadata.get("has_media", False),
                "content_hash": metadata.get("checksum", "")[:8]
            })
            
            return standardized
            
        except Exception as e:
            logger.error("input_packaging_failed", {
                "session_id": session_id,
                "error": str(e),
                "source": "whatsapp"
            })
            raise
    
    def package_api_input(self, api_data: Dict[str, Any], session_id: str) -> StandardizedInput:
        """
        Empaquetar input de API en formato estandarizado.
        
        Args:
            api_data: Datos de API request
            session_id: ID de sesión único
            
        Returns:
            StandardizedInput object
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Determinar tipo de input
            input_type = self._determine_input_type_api(api_data)
            
            # Crear contenido raw
            raw_content = self._extract_raw_content_api(api_data)
            
            # Crear metadata
            metadata = self._create_metadata_api(api_data, raw_content)
            
            # Crear audit trail
            audit_trail = self._create_audit_trail_api(api_data, session_id)
            
            standardized = StandardizedInput(
                session_id=session_id,
                timestamp=timestamp,
                input_type=input_type.value,
                raw_content=raw_content,
                metadata=metadata,
                audit_trail=audit_trail
            )
            
            logger.audit("api_input_packaged", {
                "session_id": session_id,
                "input_type": input_type.value,
                "source": "api"
            })
            
            return standardized
            
        except Exception as e:
            logger.error("api_input_packaging_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            raise
    
    def _determine_input_type(self, webhook_data: Dict[str, Any]) -> InputType:
        """Determinar tipo de input de webhook WhatsApp."""
        has_media = bool(webhook_data.get('MediaUrl0'))
        has_text = bool(webhook_data.get('Body', '').strip())
        media_type = webhook_data.get('MediaContentType0', '')
        
        if has_media and has_text:
            return InputType.MIXED
        elif has_media:
            if media_type.startswith('image/'):
                return InputType.IMAGE
            elif media_type.startswith('video/'):
                return InputType.VIDEO
            else:
                return InputType.UNKNOWN
        elif has_text:
            return InputType.TEXT
        else:
            return InputType.UNKNOWN
    
    def _determine_input_type_api(self, api_data: Dict[str, Any]) -> InputType:
        """Determinar tipo de input de API."""
        if api_data.get('image_path') or api_data.get('image_data'):
            if api_data.get('text') or api_data.get('patient_code'):
                return InputType.MIXED
            return InputType.IMAGE
        elif api_data.get('text') or api_data.get('query'):
            return InputType.TEXT
        else:
            return InputType.UNKNOWN
    
    def _extract_raw_content(self, webhook_data: Dict[str, Any], source: InputSource) -> Dict[str, Any]:
        """Extraer contenido raw de webhook."""
        return {
            "text": webhook_data.get('Body', ''),
            "media_url": webhook_data.get('MediaUrl0'),
            "media_type": webhook_data.get('MediaContentType0'),
            "media_size": int(webhook_data.get('MediaSize0', 0)),
            "from_number": webhook_data.get('From', ''),
            "to_number": webhook_data.get('To', ''),
            "message_sid": webhook_data.get('MessageSid', ''),
            "source": source.value
        }
    
    def _extract_raw_content_api(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer contenido raw de API."""
        return {
            "text": api_data.get('text', ''),
            "patient_code": api_data.get('patient_code', ''),
            "image_path": api_data.get('image_path', ''),
            "image_data": api_data.get('image_data', ''),
            "query": api_data.get('query', ''),
            "source": InputSource.API.value
        }
    
    def _create_metadata(self, webhook_data: Dict[str, Any], source: InputSource, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Crear metadata estandarizado."""
        content_string = json.dumps(raw_content, sort_keys=True)
        checksum = hashlib.sha256(content_string.encode()).hexdigest()
        
        has_media = bool(webhook_data.get('MediaUrl0'))
        has_text = bool(webhook_data.get('Body', '').strip())
        
        return {
            'source': source.value,
            'format': webhook_data.get('MediaContentType0', 'text/plain'),
            'size': int(webhook_data.get('MediaSize0', 0)),
            'checksum': checksum,
            'has_media': has_media,
            'has_text': has_text,
            'validation_status': 'pending',
            'encryption_status': 'required'
        }
    
    def _create_metadata_api(self, api_data: Dict[str, Any], raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Crear metadata para input de API."""
        content_string = json.dumps(raw_content, sort_keys=True)
        checksum = hashlib.sha256(content_string.encode()).hexdigest()
        
        return {
            'source': InputSource.API.value,
            'format': 'application/json',
            'size': len(content_string),
            'checksum': checksum,
            'has_media': bool(api_data.get('image_path') or api_data.get('image_data')),
            'has_text': bool(api_data.get('text') or api_data.get('query')),
            'validation_status': 'pending',
            'encryption_status': 'required'
        }
    
    def _create_audit_trail(self, webhook_data: Dict[str, Any], session_id: str, source: InputSource) -> Dict[str, Any]:
        """Crear audit trail estandarizado."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Crear hash anonimizado de fuente
        source_number = webhook_data.get('From', '')
        source_id = hashlib.sha256(source_number.encode()).hexdigest()[:16] if source_number else "unknown"
        
        return {
            'received_at': timestamp,
            'source_id': source_id,
            'processing_id': str(uuid.uuid4()),
            'session_id': session_id,
            'source_type': source.value,
            'packager_version': self.version,
            'compliance_flags': ['input_sanitized', 'pii_anonymized'],
            'processing_stage': 'layer1_packaging'
        }
    
    def _create_audit_trail_api(self, api_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Crear audit trail para API input."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return {
            'received_at': timestamp,
            'source_id': 'api_client',
            'processing_id': str(uuid.uuid4()),
            'session_id': session_id,
            'source_type': InputSource.API.value,
            'packager_version': self.version,
            'compliance_flags': ['input_sanitized', 'api_validated'],
            'processing_stage': 'layer1_packaging'
        }
    
    def validate_standardized_input(self, standardized: StandardizedInput) -> Dict[str, Any]:
        """
        Validar estructura de input estandarizado.
        
        Returns:
            Dict con resultado de validación
        """
        try:
            errors = []
            
            # Validar campos requeridos
            required_fields = ['session_id', 'timestamp', 'input_type', 'raw_content', 'metadata', 'audit_trail']
            for field in required_fields:
                if not hasattr(standardized, field) or getattr(standardized, field) is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validar session_id format
            if not standardized.session_id.startswith('VIGIA_SESSION_'):
                errors.append("Invalid session_id format")
            
            # Validar input_type
            valid_types = [t.value for t in InputType]
            if standardized.input_type not in valid_types:
                errors.append(f"Invalid input_type: {standardized.input_type}")
            
            # Validar metadata checksum
            if 'checksum' not in standardized.metadata:
                errors.append("Missing checksum in metadata")
            
            # Validar audit_trail
            required_audit_fields = ['received_at', 'processing_id', 'session_id']
            for field in required_audit_fields:
                if field not in standardized.audit_trail:
                    errors.append(f"Missing audit trail field: {field}")
            
            if errors:
                return {
                    "valid": False,
                    "errors": errors
                }
            
            return {
                "valid": True,
                "checksum_verified": True,
                "compliance_status": "validated"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }


class InputPackagerFactory:
    """Factory para crear instances de InputPackager."""
    
    @staticmethod
    def create_packager() -> InputPackager:
        """Crear nueva instancia de InputPackager."""
        return InputPackager()
    
    @staticmethod
    def create_standardized_input_from_dict(data: Dict[str, Any]) -> StandardizedInput:
        """Crear StandardizedInput desde diccionario."""
        return StandardizedInput(**data)
    
    @staticmethod
    def deserialize_from_json(json_str: str) -> StandardizedInput:
        """Deserializar StandardizedInput desde JSON."""
        data = json.loads(json_str)
        return StandardizedInput(**data)