"""
Production Celery Configuration for VIGIA Medical AI Pipeline
============================================================

This is the PRODUCTION-READY configuration that works with or without 
Celery installation, using fallback to mock for development/testing.

Medical-grade task configuration with HIPAA compliance considerations.
"""

import os
from typing import Dict, Any

# Configuration constants
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/1')

# Medical task configuration with clinical timeouts
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
        'max_retries': 5,   # Logging critical for HIPAA compliance
        'retry_delay': 10,
        'queue': 'audit_logging'
    },
    'notify_slack_task': {
        'time_limit': 90,   # 1.5 minutes for notifications
        'soft_time_limit': 60,
        'max_retries': 3,
        'retry_delay': 20,
        'queue': 'notifications'
    },
    'agent_coordination_task': {
        'time_limit': 300,  # 5 minutes for 9-agent coordination
        'soft_time_limit': 240,
        'max_retries': 2,
        'retry_delay': 45,
        'queue': 'medical_priority'
    }
}

def configure_task_defaults(task_name: str) -> Dict[str, Any]:
    """
    Configure defaults for medical tasks with safety-first approach.
    
    Args:
        task_name: Name of the medical task
        
    Returns:
        Dict with task configuration including medical timeouts
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

# Try to import real Celery, fallback to mock for development
try:
    from celery import Celery
    from kombu import Queue
    
    # Real Celery configuration for production
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
        
        # Medical-grade timeouts
        task_soft_time_limit=180,
        task_time_limit=300,
        
        # Retry configuration for medical reliability
        task_default_retry_delay=60,
        task_max_retries=3,
        
        # Worker configuration for medical workloads
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=100,
        
        # Monitoring for medical audit trails
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Redis configuration
        redis_max_connections=20,
        redis_retry_on_timeout=True,
        
        # Task routing for medical priorities
        task_routes={
            'src.tasks.medical.*': {'queue': 'medical_priority'},
            'src.tasks.image.*': {'queue': 'image_processing'},
            'src.tasks.notifications.*': {'queue': 'notifications'},
            'src.tasks.audit.*': {'queue': 'audit_logging'},
            'src.agents.*': {'queue': 'medical_priority'},
        },
        
        # Queue configuration for medical workflow
        task_default_queue='default',
        task_queues=(
            Queue('medical_priority', routing_key='medical'),
            Queue('image_processing', routing_key='image'),
            Queue('notifications', routing_key='notifications'),
            Queue('audit_logging', routing_key='audit'),
            Queue('default', routing_key='default'),
        ),
    )
    
    print("✅ CELERY INSTALLED: Using production medical configuration")
    CELERY_AVAILABLE = True
    
except ImportError:
    # Fallback to mock for development/testing
    from src.core.celery_mock import celery_app, MockQueue as Queue
    
    print("⚠️  CELERY NOT INSTALLED: Using mock configuration (development mode)")
    print("   For production: pip install celery==5.3.6 kombu==5.3.5")
    CELERY_AVAILABLE = False

# Auto-discover tasks for medical modules
try:
    celery_app.autodiscover_tasks([
        'src.tasks.medical',
        'src.tasks.image', 
        'src.tasks.notifications',
        'src.tasks.audit',
        'src.agents'
    ])
except Exception as e:
    print(f"⚠️  Task autodiscovery: {e}")

# Debug task for medical system verification
@celery_app.task(bind=True)
def debug_task(self):
    """Task to verify medical pipeline configuration"""
    from src.utils.secure_logger import SecureLogger
    logger = SecureLogger(__name__)
    logger.info(f'Medical pipeline request: {self.request!r}')
    return f'VIGIA Medical AI Celery configured - Redis: {REDIS_URL} - Available: {CELERY_AVAILABLE}'

if __name__ == '__main__':
    print(f"VIGIA Medical AI Celery app: {celery_app}")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Available: {CELERY_AVAILABLE}")
    celery_app.start()