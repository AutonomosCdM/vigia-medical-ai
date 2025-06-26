"""
Celery Mock for Testing Without Dependencies
==========================================

Mock implementation of Celery for testing the async pipeline without
requiring Celery installation.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock
import uuid
from datetime import datetime, timezone

class MockAsyncResult:
    """Mock AsyncResult for testing"""
    
    def __init__(self, task_id: str = None):
        self.id = task_id or str(uuid.uuid4())
        self.status = 'PENDING'
        self.result = None
        self.info = None
        
    def ready(self) -> bool:
        return self.status in ['SUCCESS', 'FAILURE', 'REVOKED']
    
    def successful(self) -> bool:
        return self.status == 'SUCCESS'
    
    def failed(self) -> bool:
        return self.status == 'FAILURE'
    
    def get(self, timeout=None, propagate=True):
        if self.status == 'SUCCESS':
            return self.result
        elif self.status == 'FAILURE':
            if propagate:
                raise Exception(self.info or "Task failed")
            return None
        else:
            raise TimeoutError("Task timeout")

class MockTask:
    """Mock Celery Task"""
    
    def __init__(self, name: str):
        self.name = name
        
    def delay(self, *args, **kwargs):
        """Mock delay method"""
        result = MockAsyncResult()
        result.status = 'SUCCESS'
        result.result = {'success': True, 'task_name': self.name}
        return result
    
    def apply_async(self, args=None, kwargs=None, **options):
        """Mock apply_async method"""
        return self.delay(*(args or []), **(kwargs or {}))

class MockCeleryApp:
    """Mock Celery App"""
    
    def __init__(self, name: str):
        self.name = name
        self.conf = MagicMock()
        self.control = MagicMock()
        
        # Mock inspect
        self.control.inspect.return_value.active_queues.return_value = {}
        self.control.inspect.return_value.stats.return_value = {}
        self.control.inspect.return_value.active.return_value = {}
        self.control.inspect.return_value.scheduled.return_value = {}
        self.control.inspect.return_value.reserved.return_value = {}
        self.control.ping.return_value = ['pong']
        
    def AsyncResult(self, task_id: str):
        """Mock AsyncResult creation"""
        result = MockAsyncResult(task_id)
        result.status = 'SUCCESS'
        result.result = {'success': True, 'task_id': task_id}
        return result
    
    def task(self, *args, **kwargs):
        """Mock task decorator"""
        def decorator(func):
            task = MockTask(func.__name__)
            task.delay = lambda *a, **kw: MockAsyncResult(f"{func.__name__}_{uuid.uuid4()}")
            return task
        return decorator
    
    def autodiscover_tasks(self, packages):
        """Mock autodiscover"""
        pass

# Mock objects for imports
class MockQueue:
    def __init__(self, name, routing_key=None):
        self.name = name
        self.routing_key = routing_key

class MockGroup:
    """Mock Celery group"""
    
    def __init__(self, tasks):
        self.tasks = tasks
        
    def get(self, timeout=None, propagate=True):
        """Mock group get with results"""
        return [{'success': True} for _ in self.tasks]

# Export mock functions
def group(tasks):
    """Mock group function"""
    return MockGroup(tasks)

# Mock celery app instance
celery_app = MockCeleryApp('vigia_medical_pipeline')

# Mock configuration
MEDICAL_TASK_CONFIG = {
    'image_analysis_task': {
        'time_limit': 240,
        'soft_time_limit': 180,
        'max_retries': 2,
        'retry_delay': 30,
        'queue': 'image_processing'
    }
}

def configure_task_defaults(task_name: str):
    """Mock configure_task_defaults"""
    return MEDICAL_TASK_CONFIG.get(task_name, {
        'time_limit': 300,
        'soft_time_limit': 240,
        'max_retries': 3,
        'retry_delay': 60,
        'queue': 'default'
    })