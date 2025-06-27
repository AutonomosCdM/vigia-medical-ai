"""
Celery Configuration for Vigia Medical Pipeline
==============================================

Configuración de Celery para tareas asíncronas médicas con Redis backend.
Implementa retry policies y timeouts específicos para el contexto médico.
"""

import os
from celery import Celery
from kombu import Queue
from src.utils.secure_logger import SecureLogger

# Configuración de Redis backend
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/1')

# Inicializar Celery app
celery_app = Celery('vigia_medical_pipeline')

# Configuración del broker y backend
celery_app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    
    # Serialización
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Santiago',
    enable_utc=True,
    
    # Timeouts específicos para contexto médico
    task_soft_time_limit=180,  # 3 minutos warning
    task_time_limit=300,       # 5 minutos hard limit
    
    # Retry configuration
    task_default_retry_delay=60,    # 1 minuto entre reintentos
    task_max_retries=3,             # Máximo 3 reintentos para seguridad médica
    
    # Worker configuration
    worker_prefetch_multiplier=1,   # Un task por vez para procesos médicos
    worker_max_tasks_per_child=100, # Reciclar workers para estabilidad
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Redis configuration
    redis_max_connections=20,
    redis_retry_on_timeout=True,
    
    # Task routing
    task_routes={
        'vigia_detect.tasks.medical.*': {'queue': 'medical_priority'},
        'vigia_detect.tasks.image.*': {'queue': 'image_processing'},
        'vigia_detect.tasks.notifications.*': {'queue': 'notifications'},
        'vigia_detect.tasks.audit.*': {'queue': 'audit_logging'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('medical_priority', routing_key='medical'),
        Queue('image_processing', routing_key='image'),
        Queue('notifications', routing_key='notifications'),
        Queue('audit_logging', routing_key='audit'),
        Queue('default', routing_key='default'),
    ),
)

# Configuración específica de tareas médicas
MEDICAL_TASK_CONFIG = {
    'image_analysis_task': {
        'time_limit': 240,  # 4 minutos para análisis de imagen
        'soft_time_limit': 180,
        'max_retries': 2,
        'retry_delay': 30,
        'queue': 'image_processing'
    },
    'risk_score_task': {
        'time_limit': 120,  # 2 minutos para scoring
        'soft_time_limit': 90,
        'max_retries': 3,
        'retry_delay': 15,
        'queue': 'medical_priority'
    },
    'audit_log_task': {
        'time_limit': 60,   # 1 minuto para logging
        'soft_time_limit': 45,
        'max_retries': 5,   # Logging crítico para compliance
        'retry_delay': 10,
        'queue': 'audit_logging'
    },
    'notify_slack_task': {
        'time_limit': 90,   # 1.5 minutos para notificaciones
        'soft_time_limit': 60,
        'max_retries': 3,
        'retry_delay': 20,
        'queue': 'notifications'
    }
}

# Logger seguro para Celery
logger = SecureLogger(__name__)

def configure_task_defaults(task_name: str):
    """
    Configura defaults específicos para tareas médicas
    
    Args:
        task_name: Nombre de la tarea médica
        
    Returns:
        Dict con configuración de la tarea
    """
    config = MEDICAL_TASK_CONFIG.get(task_name, {})
    
    # Defaults seguros para tareas médicas
    default_config = {
        'time_limit': 300,
        'soft_time_limit': 240,
        'max_retries': 3,
        'retry_delay': 60,
        'queue': 'default'
    }
    
    # Merge con configuración específica
    return {**default_config, **config}

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'vigia_detect.tasks.medical',
    'vigia_detect.tasks.image', 
    'vigia_detect.tasks.notifications',
    'vigia_detect.tasks.audit'
])

# Configuración de logging
@celery_app.task(bind=True)
def debug_task(self):
    """Task de debug para verificar configuración"""
    logger.info(f'Request: {self.request!r}')
    return f'Celery configured successfully - Redis: {REDIS_URL}'

if __name__ == '__main__':
    celery_app.start()