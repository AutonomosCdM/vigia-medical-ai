"""
Production Celery Configuration for Vigia Medical Pipeline
==========================================================

This is the PRODUCTION-READY configuration that works with or without 
Celery installation, using fallback to mock for development/testing.
"""

import os
from typing import Dict, Any

# Configuration constants
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/1')

# Medical task configuration
MEDICAL_TASK_CONFIG = {
    'image_analysis_task': {
        'time_limit': 240,  # 4 minutes for image analysis
        'soft_time_limit': 180,
        'max_retries': 2,
        'retry_delay': 30,
        'queue': 'image_processing'
    },
    'risk_score_task': {
        'time_limit': 120,  # 2 minutes for scoring
        'soft_time_limit': 90,
        'max_retries': 3,
        'retry_delay': 15,
        'queue': 'medical_priority'
    },
    'audit_log_task': {
        'time_limit': 60,   # 1 minute for logging
        'soft_time_limit': 45,
        'max_retries': 5,   # Logging critical for compliance
        'retry_delay': 10,
        'queue': 'audit_logging'
    },
    'notify_slack_task': {
        'time_limit': 90,   # 1.5 minutes for notifications
        'soft_time_limit': 60,
        'max_retries': 3,
        'retry_delay': 20,
        'queue': 'notifications'
    }
}

def configure_task_defaults(task_name: str) -> Dict[str, Any]:
    """
    Configure defaults for medical tasks
    
    Args:
        task_name: Name of the medical task
        
    Returns:
        Dict with task configuration
    """
    config = MEDICAL_TASK_CONFIG.get(task_name, {})
    
    # Safe defaults for medical tasks
    default_config = {
        'time_limit': 300,
        'soft_time_limit': 240,
        'max_retries': 3,
        'retry_delay': 60,
        'queue': 'default'
    }
    
    return {**default_config, **config}

# Try to import real Celery, fallback to mock
try:
    from celery import Celery
    from kombu import Queue
    
    # Real Celery configuration
    celery_app = Celery('vigia_medical_pipeline')
    
    celery_app.conf.update(
        broker_url=REDIS_URL,
        result_backend=REDIS_URL,
        
        # Serialization
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='America/Santiago',
        enable_utc=True,
        
        # Medical timeouts
        task_soft_time_limit=180,
        task_time_limit=300,
        
        # Retry configuration
        task_default_retry_delay=60,
        task_max_retries=3,
        
        # Worker configuration
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=100,
        
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
    
    print("✅ CELERY INSTALLED: Using production configuration")
    CELERY_AVAILABLE = True
    
except ImportError:
    # Fallback to mock for development/testing
    from vigia_detect.core.celery_mock import celery_app, MockQueue as Queue
    
    print("⚠️  CELERY NOT INSTALLED: Using mock configuration (development mode)")
    print("   For production: pip install celery==5.3.6 kombu==5.3.5")
    CELERY_AVAILABLE = False

# Auto-discover tasks (works with both real and mock)
try:
    celery_app.autodiscover_tasks([
        'vigia_detect.tasks.medical',
        'vigia_detect.tasks.image', 
        'vigia_detect.tasks.notifications',
        'vigia_detect.tasks.audit'
    ])
except Exception as e:
    print(f"⚠️  Task autodiscovery: {e}")

# Debug task for verification
@celery_app.task(bind=True)
def debug_task(self):
    """Task to verify configuration"""
    from vigia_detect.utils.secure_logger import SecureLogger
    logger = SecureLogger(__name__)
    logger.info(f'Request: {self.request!r}')
    return f'Celery configured successfully - Redis: {REDIS_URL} - Available: {CELERY_AVAILABLE}'

if __name__ == '__main__':
    print(f"Celery app: {celery_app}")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Available: {CELERY_AVAILABLE}")
    celery_app.start()