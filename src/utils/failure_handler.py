"""
Failure Handler for Asynchronous Medical Tasks
==============================================

Manejo centralizado de fallos en tareas asíncronas médicas con logging seguro,
escalación automática y notificaciones críticas para garantizar seguridad del paciente.
"""

import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from vigia_detect.utils.secure_logger import SecureLogger
from vigia_detect.utils.audit_service import AuditService

class FailureSeverity(Enum):
    """Niveles de severidad para fallos médicos"""
    LOW = "low"                    # Fallos no críticos
    MEDIUM = "medium"             # Fallos que afectan workflow
    HIGH = "high"                 # Fallos que impactan análisis médico
    CRITICAL = "critical"         # Fallos que comprometen seguridad del paciente

class TaskFailureHandler:
    """
    Manejador centralizado de fallos en tareas asíncronas médicas
    """
    
    def __init__(self):
        self.logger = SecureLogger(__name__)
        self.audit_service = AuditService()
        
        # Configuración de severidad por tipo de task
        self.task_severity_map = {
            'image_analysis_task': FailureSeverity.HIGH,
            'risk_score_task': FailureSeverity.CRITICAL,
            'audit_log_task': FailureSeverity.MEDIUM,
            'notify_slack_task': FailureSeverity.LOW,
            'medical_analysis_task': FailureSeverity.CRITICAL,
            'triage_task': FailureSeverity.CRITICAL
        }
        
        # Configuración de escalación por severidad
        self.escalation_config = {
            FailureSeverity.LOW: {
                'retry_attempts': 3,
                'escalate_after': 5,
                'notification_required': False
            },
            FailureSeverity.MEDIUM: {
                'retry_attempts': 3,
                'escalate_after': 3,
                'notification_required': True
            },
            FailureSeverity.HIGH: {
                'retry_attempts': 2,
                'escalate_after': 2,
                'notification_required': True
            },
            FailureSeverity.CRITICAL: {
                'retry_attempts': 1,
                'escalate_after': 1,
                'notification_required': True
            }
        }

    def handle_task_failure(
        self, 
        task_name: str,
        task_id: str,
        exception: Exception,
        traceback_str: str,
        context: Optional[Dict[str, Any]] = None,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Maneja fallo de tarea asíncrona con logging y escalación
        
        Args:
            task_name: Nombre de la tarea que falló
            task_id: ID único de la tarea
            exception: Excepción capturada
            traceback_str: Stack trace del error
            context: Contexto adicional del error
            patient_context: Contexto del paciente (si aplica)
            
        Returns:
            Dict con información de manejo del fallo
        """
        severity = self._determine_severity(task_name)
        failure_time = datetime.now(timezone.utc)
        
        # Log seguro del fallo
        failure_data = {
            'task_name': task_name,
            'task_id': task_id,
            'severity': severity.value,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'failure_time': failure_time.isoformat(),
            'context': context or {},
            'has_patient_context': patient_context is not None
        }
        
        # Log según severidad
        if severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL]:
            self.logger.error(
                f"MEDICAL TASK FAILURE: {task_name}",
                extra=failure_data,
                exc_info=True
            )
        else:
            self.logger.warning(
                f"Task failure: {task_name}",
                extra=failure_data
            )
        
        # Auditoría para compliance médico
        self._log_medical_audit(failure_data, patient_context)
        
        # Determinar acción según severidad
        escalation_response = self._determine_escalation_action(
            severity, task_name, failure_data
        )
        
        # Notificación si requerida
        if self._requires_notification(severity):
            self._send_failure_notification(failure_data, escalation_response)
        
        return {
            'failure_logged': True,
            'severity': severity.value,
            'escalation_response': escalation_response,
            'audit_logged': True,
            'notification_sent': self._requires_notification(severity)
        }
    
    def handle_retry_exhausted(
        self,
        task_name: str,
        task_id: str,
        retry_count: int,
        final_exception: Exception,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Maneja agotamiento de reintentos en tareas médicas críticas
        
        Args:
            task_name: Nombre de la tarea
            task_id: ID de la tarea
            retry_count: Número de reintentos realizados
            final_exception: Última excepción
            patient_context: Contexto del paciente
            
        Returns:
            Dict con respuesta de escalación
        """
        severity = self._determine_severity(task_name)
        
        # Log crítico para agotamiento de reintentos
        self.logger.critical(
            f"RETRY EXHAUSTED - MEDICAL TASK: {task_name}",
            extra={
                'task_id': task_id,
                'retry_count': retry_count,
                'final_exception': str(final_exception),
                'severity': severity.value,
                'requires_manual_intervention': True
            }
        )
        
        # Escalación automática para tareas críticas
        if severity == FailureSeverity.CRITICAL:
            return self._escalate_to_human_review(
                task_name, task_id, patient_context
            )
        
        return {
            'escalated': False,
            'requires_manual_review': True,
            'severity': severity.value
        }
    
    def _determine_severity(self, task_name: str) -> FailureSeverity:
        """Determina severidad del fallo según el tipo de tarea"""
        return self.task_severity_map.get(task_name, FailureSeverity.MEDIUM)
    
    def _log_medical_audit(
        self, 
        failure_data: Dict[str, Any], 
        patient_context: Optional[Dict[str, Any]]
    ) -> None:
        """Log de auditoría médica para compliance"""
        try:
            audit_data = {
                'event_type': 'task_failure',
                'task_name': failure_data['task_name'],
                'severity': failure_data['severity'],
                'timestamp': failure_data['failure_time'],
                'requires_review': failure_data['severity'] in ['high', 'critical']
            }
            
            # Agregar contexto del paciente (si existe) de forma segura
            if patient_context:
                audit_data['patient_id'] = patient_context.get('patient_code', 'unknown')
                audit_data['has_medical_context'] = True
            
            self.audit_service.log_system_event(audit_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log audit for task failure: {e}")
    
    def _determine_escalation_action(
        self, 
        severity: FailureSeverity, 
        task_name: str, 
        failure_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determina acción de escalación según severidad"""
        config = self.escalation_config[severity]
        
        escalation = {
            'requires_escalation': severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL],
            'max_retries': config['retry_attempts'],
            'escalate_after_failures': config['escalate_after'],
            'urgency_level': 'immediate' if severity == FailureSeverity.CRITICAL else 'normal'
        }
        
        if severity == FailureSeverity.CRITICAL:
            escalation.update({
                'requires_human_review': True,
                'requires_medical_attention': True,
                'fallback_procedure': 'manual_processing_required'
            })
        
        return escalation
    
    def _requires_notification(self, severity: FailureSeverity) -> bool:
        """Determina si el fallo requiere notificación"""
        return self.escalation_config[severity]['notification_required']
    
    def _send_failure_notification(
        self, 
        failure_data: Dict[str, Any], 
        escalation_response: Dict[str, Any]
    ) -> None:
        """Envía notificación de fallo (placeholder para integración con Slack)"""
        try:
            # TODO: Integrar con notify_slack_task cuando esté disponible
            self.logger.warning(
                "NOTIFICATION REQUIRED - Task failure",
                extra={
                    'task_name': failure_data['task_name'],
                    'severity': failure_data['severity'],
                    'requires_escalation': escalation_response['requires_escalation']
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to send failure notification: {e}")
    
    def _escalate_to_human_review(
        self, 
        task_name: str, 
        task_id: str, 
        patient_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Escalación a revisión humana para tareas críticas"""
        escalation_data = {
            'escalated': True,
            'escalation_type': 'human_review_required',
            'escalation_time': datetime.now(timezone.utc).isoformat(),
            'task_name': task_name,
            'task_id': task_id,
            'priority': 'critical',
            'requires_immediate_attention': True
        }
        
        if patient_context:
            escalation_data['patient_affected'] = True
            escalation_data['patient_code'] = patient_context.get('patient_code', 'unknown')
        
        # Log de escalación crítica
        self.logger.critical(
            "ESCALATED TO HUMAN REVIEW - Medical task failure",
            extra=escalation_data
        )
        
        return escalation_data

# Instancia global del manejador
failure_handler = TaskFailureHandler()

def log_task_failure(
    task_name: str,
    task_id: str, 
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    patient_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Función helper para logging de fallos de tareas
    
    Args:
        task_name: Nombre de la tarea que falló
        task_id: ID de la tarea
        exception: Excepción capturada
        context: Contexto adicional
        patient_context: Contexto del paciente
        
    Returns:
        Dict con resultado del manejo del fallo
    """
    return failure_handler.handle_task_failure(
        task_name=task_name,
        task_id=task_id,
        exception=exception,
        traceback_str=traceback.format_exc(),
        context=context,
        patient_context=patient_context
    )