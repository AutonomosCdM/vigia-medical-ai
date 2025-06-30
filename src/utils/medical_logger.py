"""
Medical Logging System - VIGIA Medical AI
=========================================

HIPAA-compliant unified logging system for medical operations
with audit trail and sensitive data protection.
"""

import logging
import logging.handlers
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from config.settings import settings


class MedicalLogLevel(Enum):
    """Medical-specific log levels with clinical context."""
    PATIENT_EVENT = "PATIENT_EVENT"         # Patient interactions
    MEDICAL_DECISION = "MEDICAL_DECISION"   # Clinical decisions
    HIPAA_AUDIT = "HIPAA_AUDIT"            # HIPAA compliance events
    SECURITY_EVENT = "SECURITY_EVENT"       # Security-related events
    SYSTEM_ERROR = "SYSTEM_ERROR"          # System errors affecting patient care
    PERFORMANCE = "PERFORMANCE"            # Performance monitoring


@dataclass
class MedicalLogEntry:
    """Structured medical log entry with HIPAA compliance."""
    timestamp: str
    level: str
    event_type: MedicalLogLevel
    batman_token: Optional[str]  # PHI tokenized patient identifier
    session_id: Optional[str]
    component: str
    operation: str
    message: str
    metadata: Dict[str, Any]
    duration_ms: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    hipaa_category: str = "GENERAL"
    
    def __post_init__(self):
        """Validate and sanitize log entry."""
        # Ensure no PHI in message or metadata
        self._sanitize_phi()
        
        # Set HIPAA category if not specified
        if self.hipaa_category == "GENERAL":
            self._set_hipaa_category()
    
    def _sanitize_phi(self) -> None:
        """Remove potential PHI from log data."""
        # List of potential PHI patterns to mask
        phi_patterns = [
            'ssn', 'social_security', 'phone', 'email', 'address',
            'name', 'birth_date', 'mrn', 'patient_id'
        ]
        
        # Sanitize message
        for pattern in phi_patterns:
            if pattern in self.message.lower():
                # Replace with generic identifier
                self.message = self.message.replace(pattern, f"[{pattern.upper()}_MASKED]")
        
        # Sanitize metadata
        if isinstance(self.metadata, dict):
            for key in list(self.metadata.keys()):
                if any(pattern in key.lower() for pattern in phi_patterns):
                    # Hash the value instead of storing directly
                    original_value = str(self.metadata[key])
                    self.metadata[key] = f"HASHED_{hashlib.sha256(original_value.encode()).hexdigest()[:8]}"
    
    def _set_hipaa_category(self) -> None:
        """Set HIPAA category based on event type."""
        category_mapping = {
            MedicalLogLevel.PATIENT_EVENT: "PHI_ACCESS",
            MedicalLogLevel.MEDICAL_DECISION: "CLINICAL_DECISION",
            MedicalLogLevel.HIPAA_AUDIT: "AUDIT_LOG",
            MedicalLogLevel.SECURITY_EVENT: "SECURITY",
            MedicalLogLevel.SYSTEM_ERROR: "SYSTEM_INTEGRITY",
            MedicalLogLevel.PERFORMANCE: "SYSTEM_PERFORMANCE"
        }
        self.hipaa_category = category_mapping.get(self.event_type, "GENERAL")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data


class MedicalLogger:
    """
    HIPAA-compliant medical logging system.
    
    Features:
    - PHI tokenization support (Batman tokens)
    - Structured medical event logging
    - Audit trail compliance
    - Performance monitoring
    - Sensitive data sanitization
    """
    
    def __init__(self, component_name: str):
        """
        Initialize medical logger for specific component.
        
        Args:
            component_name: Name of the component (e.g., 'risk_assessment', 'image_analysis')
        """
        self.component_name = component_name
        self.logger = self._setup_logger()
        self.audit_logger = self._setup_audit_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup standard medical logger."""
        logger = logging.getLogger(f"vigia.medical.{self.component_name}")
        
        if not logger.handlers:
            # Console Handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - VIGIA-MEDICAL - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File Handler for Medical Logs
            logs_dir = Path("logs/medical")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                logs_dir / f"{self.component_name}.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Set level from settings
            logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        return logger
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup HIPAA audit logger."""
        audit_logger = logging.getLogger(f"vigia.audit.{self.component_name}")
        
        if not audit_logger.handlers and settings.hipaa_settings.AUDIT_LOGGING_ENABLED:
            # Audit File Handler (separate from regular logs)
            audit_dir = Path("logs/audit")
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_dir / f"hipaa_audit_{self.component_name}.log",
                maxBytes=50*1024*1024,  # 50MB for audit logs
                backupCount=20,  # Keep more audit logs
                encoding='utf-8'
            )
            
            # JSON formatter for structured audit logs
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_obj = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'component': record.name,
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'audit_type': 'HIPAA_MEDICAL'
                    }
                    return json.dumps(log_obj)
            
            audit_handler.setFormatter(JSONFormatter())
            audit_logger.addHandler(audit_handler)
            audit_logger.setLevel(logging.INFO)
        
        return audit_logger
    
    def log_medical_event(self, 
                         event_type: MedicalLogLevel,
                         operation: str,
                         message: str,
                         batman_token: Optional[str] = None,
                         session_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         duration_ms: Optional[float] = None,
                         error_details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a medical event with HIPAA compliance.
        
        Args:
            event_type: Type of medical event
            operation: Operation being performed
            message: Log message (will be sanitized)
            batman_token: PHI tokenized patient identifier
            session_id: Session identifier
            metadata: Additional metadata (will be sanitized)
            duration_ms: Operation duration in milliseconds
            error_details: Error details if applicable
        """
        entry = MedicalLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level="INFO",
            event_type=event_type,
            batman_token=batman_token,
            session_id=session_id,
            component=self.component_name,
            operation=operation,
            message=message,
            metadata=metadata or {},
            duration_ms=duration_ms,
            error_details=error_details
        )
        
        # Log to standard logger
        self.logger.info(f"[{event_type.value}] {operation}: {message}")
        
        # Log to audit system if enabled
        if settings.hipaa_settings.AUDIT_LOGGING_ENABLED:
            self.audit_logger.info(json.dumps(entry.to_dict()))
    
    def log_patient_interaction(self, 
                              batman_token: str,
                              operation: str,
                              outcome: str,
                              session_id: Optional[str] = None) -> None:
        """Log patient interaction for HIPAA audit."""
        self.log_medical_event(
            event_type=MedicalLogLevel.PATIENT_EVENT,
            operation=operation,
            message=f"Patient interaction: {outcome}",
            batman_token=batman_token,
            session_id=session_id,
            metadata={"outcome": outcome}
        )
    
    def log_medical_decision(self,
                           batman_token: str,
                           decision_type: str,
                           recommendation: str,
                           evidence_level: str,
                           confidence: float,
                           session_id: Optional[str] = None) -> None:
        """Log medical decision for clinical audit."""
        self.log_medical_event(
            event_type=MedicalLogLevel.MEDICAL_DECISION,
            operation="clinical_decision",
            message=f"Medical decision: {decision_type}",
            batman_token=batman_token,
            session_id=session_id,
            metadata={
                "decision_type": decision_type,
                "recommendation": recommendation,
                "evidence_level": evidence_level,
                "confidence": confidence
            }
        )
    
    def log_performance(self,
                       operation: str,
                       duration_ms: float,
                       success: bool,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics."""
        status = "SUCCESS" if success else "FAILURE"
        self.log_medical_event(
            event_type=MedicalLogLevel.PERFORMANCE,
            operation=operation,
            message=f"Performance: {operation} completed in {duration_ms:.2f}ms - {status}",
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
    
    def log_error(self,
                  operation: str,
                  error: Exception,
                  batman_token: Optional[str] = None,
                  session_id: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Log system error affecting medical operations."""
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.log_medical_event(
            event_type=MedicalLogLevel.SYSTEM_ERROR,
            operation=operation,
            message=f"System error in {operation}: {type(error).__name__}",
            batman_token=batman_token,
            session_id=session_id,
            error_details=error_details
        )
        
        # Also log to standard error level
        self.logger.error(f"Medical system error in {operation}", exc_info=True)


# Singleton pattern for global medical logger
_medical_loggers: Dict[str, MedicalLogger] = {}


def get_medical_logger(component_name: str) -> MedicalLogger:
    """
    Get or create a medical logger for the specified component.
    
    Args:
        component_name: Component identifier
        
    Returns:
        MedicalLogger instance
    """
    if component_name not in _medical_loggers:
        _medical_loggers[component_name] = MedicalLogger(component_name)
    
    return _medical_loggers[component_name]


# Convenience functions
def log_patient_event(component: str, batman_token: str, operation: str, outcome: str) -> None:
    """Quick patient event logging."""
    logger = get_medical_logger(component)
    logger.log_patient_interaction(batman_token, operation, outcome)


def log_medical_decision(component: str, batman_token: str, decision_type: str, 
                        recommendation: str, evidence_level: str, confidence: float) -> None:
    """Quick medical decision logging."""
    logger = get_medical_logger(component)
    logger.log_medical_decision(batman_token, decision_type, recommendation, 
                              evidence_level, confidence)


def log_performance(component: str, operation: str, duration_ms: float, success: bool) -> None:
    """Quick performance logging."""
    logger = get_medical_logger(component)
    logger.log_performance(operation, duration_ms, success)