"""
Enhanced error handling utilities for the Vigia medical detection system.
Provides medical-specific error handling and user-friendly error messages.
"""
import sys
import traceback
from enum import Enum
from typing import Dict, Any, Optional, Type, Union, List
from datetime import datetime
import logging

from .shared_utilities import VigiaLogger


class ErrorSeverity(Enum):
    """Error severity levels for medical systems"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in the medical system"""
    VALIDATION = "validation"
    PROCESSING = "processing"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    MEDICAL_DATA = "medical_data"
    FILE_SYSTEM = "file_system"


class VigiaError(Exception):
    """
    Base exception class for Vigia medical detection system.
    Provides structured error information for medical applications.
    """
    
    def __init__(self,
                 message: str,
                 error_code: str,
                 category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 user_message: Optional[str] = None,
                 recovery_suggestions: Optional[List[str]] = None):
        """
        Initialize Vigia error.
        
        Args:
            message: Technical error message for developers
            error_code: Unique error code for tracking
            category: Error category for classification
            severity: Error severity level
            context: Additional context information
            user_message: User-friendly error message
            recovery_suggestions: List of recovery suggestions
        """
        super().__init__(message)
        
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.user_message = user_message or self._generate_user_message()
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()
        self.error_id = self._generate_error_id()
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        return f"VIG-{self.timestamp.strftime('%Y%m%d-%H%M%S')}-{hash(self.message) % 10000:04d}"
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message based on category"""
        user_messages = {
            ErrorCategory.VALIDATION: "Los datos proporcionados no son válidos.",
            ErrorCategory.PROCESSING: "Error en el procesamiento de la imagen médica.",
            ErrorCategory.NETWORK: "Error de conexión de red.",
            ErrorCategory.DATABASE: "Error en la base de datos del sistema.",
            ErrorCategory.AUTHENTICATION: "Error de autenticación.",
            ErrorCategory.CONFIGURATION: "Error de configuración del sistema.",
            ErrorCategory.EXTERNAL_API: "Error en servicio externo.",
            ErrorCategory.MEDICAL_DATA: "Error en los datos médicos.",
            ErrorCategory.FILE_SYSTEM: "Error en el sistema de archivos."
        }
        return user_messages.get(self.category, "Se ha producido un error en el sistema.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": str(self),
            "user_message": self.user_message,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc() if sys.exc_info()[0] else None
        }


class PatientDataError(VigiaError):
    """Error related to patient data validation or processing"""
    
    def __init__(self, message: str, patient_code: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="PATIENT_DATA_ERROR",
            category=ErrorCategory.MEDICAL_DATA,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if patient_code:
            self.context["patient_code"] = patient_code


class ImageProcessingError(VigiaError):
    """Error related to medical image processing"""
    
    def __init__(self, message: str, image_path: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="IMAGE_PROCESSING_ERROR",
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if image_path:
            self.context["image_path"] = image_path


class DetectionError(VigiaError):
    """Error related to lesion detection"""
    
    def __init__(self, message: str, detection_stage: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="DETECTION_ERROR",
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if detection_stage:
            self.context["detection_stage"] = detection_stage


class DatabaseError(VigiaError):
    """Error related to database operations"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if operation:
            self.context["operation"] = operation


class ExternalAPIError(VigiaError):
    """Error related to external API calls"""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="EXTERNAL_API_ERROR",
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if api_name:
            self.context["api_name"] = api_name
        if status_code:
            self.context["status_code"] = status_code


class MedicalErrorHandler:
    """
    Enhanced error handler specifically designed for medical applications.
    Provides HIPAA-compliant logging and error tracking.
    """
    
    def __init__(self, module_name: str):
        self.logger = VigiaLogger.get_logger(f"{module_name}.errors")
        self.module_name = module_name
    
    def handle_error(self,
                    error: Union[Exception, VigiaError],
                    operation: str,
                    context: Optional[Dict[str, Any]] = None,
                    suppress_sensitive_data: bool = True) -> Dict[str, Any]:
        """
        Handle an error with medical-specific considerations.
        
        Args:
            error: The error that occurred
            operation: Description of the operation that failed
            context: Additional context (will be sanitized)
            suppress_sensitive_data: Whether to suppress sensitive medical data
            
        Returns:
            Structured error response
        """
        # Sanitize context to remove sensitive medical data
        sanitized_context = self._sanitize_context(context) if suppress_sensitive_data else context
        
        if isinstance(error, VigiaError):
            # Already a structured Vigia error
            error_dict = error.to_dict()
            error_dict["operation"] = operation
            if sanitized_context:
                error_dict["context"].update(sanitized_context)
        else:
            # Convert to VigiaError
            vigia_error = self._convert_to_vigia_error(error, operation)
            error_dict = vigia_error.to_dict()
            if sanitized_context:
                error_dict["context"].update(sanitized_context)
        
        # Log the error based on severity
        self._log_error(error_dict)
        
        # Create response for the caller
        return {
            "success": False,
            "error_id": error_dict["error_id"],
            "user_message": error_dict["user_message"],
            "recovery_suggestions": error_dict["recovery_suggestions"],
            "severity": error_dict["severity"],
            "category": error_dict["category"],
            "timestamp": error_dict["timestamp"]
        }
    
    def _sanitize_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sanitize context to remove sensitive medical data.
        
        Args:
            context: Context dictionary that may contain sensitive data
            
        Returns:
            Sanitized context dictionary
        """
        if not context:
            return {}
        
        # Fields that should be removed or masked for HIPAA compliance
        sensitive_fields = {
            "patient_name", "patient_id", "ssn", "medical_record_number",
            "phone_number", "email", "address", "date_of_birth"
        }
        
        sanitized = {}
        for key, value in context.items():
            if key.lower() in sensitive_fields:
                # Mask sensitive data
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 100:
                # Truncate very long strings that might contain sensitive data
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _convert_to_vigia_error(self, error: Exception, operation: str) -> VigiaError:
        """Convert standard exception to VigiaError"""
        # Map common exceptions to appropriate categories
        error_mapping = {
            ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
            FileNotFoundError: (ErrorCategory.FILE_SYSTEM, ErrorSeverity.MEDIUM),
            PermissionError: (ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH),
            ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
        }
        
        category, severity = error_mapping.get(type(error), (ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM))
        
        return VigiaError(
            message=str(error),
            error_code=f"{type(error).__name__.upper()}",
            category=category,
            severity=severity,
            context={"operation": operation, "original_type": type(error).__name__}
        )
    
    def _log_error(self, error_dict: Dict[str, Any]) -> None:
        """Log error based on severity level"""
        severity = error_dict["severity"]
        error_message = (
            f"Error ID: {error_dict['error_id']} | "
            f"Category: {error_dict['category']} | "
            f"Message: {error_dict['message']}"
        )
        
        if severity == ErrorSeverity.CRITICAL.value:
            self.logger.critical(error_message, extra={"error_data": error_dict})
        elif severity == ErrorSeverity.HIGH.value:
            self.logger.error(error_message, extra={"error_data": error_dict})
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(error_message, extra={"error_data": error_dict})
        else:
            self.logger.info(error_message, extra={"error_data": error_dict})
    
    def create_recovery_suggestions(self, error_category: ErrorCategory) -> List[str]:
        """
        Create recovery suggestions based on error category.
        
        Args:
            error_category: The category of error
            
        Returns:
            List of recovery suggestions
        """
        suggestions = {
            ErrorCategory.VALIDATION: [
                "Verifique que los datos ingresados sean correctos",
                "Revise el formato de los campos requeridos",
                "Contacte al administrador si el problema persiste"
            ],
            ErrorCategory.PROCESSING: [
                "Intente procesar la imagen nuevamente",
                "Verifique que la imagen sea válida y esté en formato correcto",
                "Contacte al soporte técnico si el error continúa"
            ],
            ErrorCategory.NETWORK: [
                "Verifique su conexión a internet",
                "Intente nuevamente en unos minutos",
                "Contacte al administrador de red si el problema persiste"
            ],
            ErrorCategory.DATABASE: [
                "Intente la operación nuevamente",
                "Contacte al administrador de base de datos",
                "Verifique que el sistema esté disponible"
            ],
            ErrorCategory.MEDICAL_DATA: [
                "Verifique que los datos del paciente sean correctos",
                "Asegúrese de tener los permisos necesarios",
                "Contacte al responsable de datos médicos"
            ]
        }
        
        return suggestions.get(error_category, [
            "Intente la operación nuevamente",
            "Contacte al soporte técnico si el problema persiste"
        ])


def handle_exceptions(default_return=None):
    """
    Decorator para manejar excepciones de manera consistente.
    
    Args:
        default_return: Valor a retornar en caso de excepción
        
    Returns:
        Función decorada con manejo de excepciones
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log del error
                logger = logging.getLogger(func.__module__)
                logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                
                # Retornar valor por defecto
                return default_return
                
        # Para funciones async
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log del error
                logger = logging.getLogger(func.__module__)
                logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                
                # Retornar valor por defecto
                return default_return
        
        # Retornar el wrapper apropiado
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
            
    return decorator