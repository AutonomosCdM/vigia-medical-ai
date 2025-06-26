"""
Service Configuration Manager for Vigia Medical AI System
========================================================

Automatically detects and configures real services vs mocks based on environment.
Supports development (real services) and testing (mock services) configurations.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ServiceMode(Enum):
    """Service execution modes"""
    MOCK = "mock"
    REAL = "real"
    AUTO = "auto"


class ServiceType(Enum):
    """Available service types"""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGING = "messaging"
    STORAGE = "storage"
    AI_MODEL = "ai_model"
    CELERY = "celery"


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    service_type: ServiceType
    mode: ServiceMode
    config: Dict[str, Any]
    available: bool = False
    health_check_url: Optional[str] = None


class ServiceConfigManager:
    """Manages service configurations for real vs mock services"""
    
    def __init__(self, force_mode: Optional[ServiceMode] = None):
        """
        Initialize service configuration manager
        
        Args:
            force_mode: Force a specific mode (override auto-detection)
        """
        self.force_mode = force_mode
        self.configs: Dict[ServiceType, ServiceConfig] = {}
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.is_testing = 'pytest' in os.environ.get('_', '') or 'test' in os.environ.get('PYTEST_CURRENT_TEST', '')
        
        self._detect_and_configure_services()
    
    def _detect_and_configure_services(self):
        """Detect available services and configure appropriately"""
        logger.info(f"Configuring services for environment: {self.environment}")
        
        # Determine default mode
        if self.force_mode:
            default_mode = self.force_mode
        elif self.is_testing:
            default_mode = ServiceMode.MOCK
        elif self.environment in ['test', 'ci']:
            default_mode = ServiceMode.MOCK
        else:
            default_mode = ServiceMode.REAL
        
        logger.info(f"Default service mode: {default_mode.value}")
        
        # Configure each service
        self._configure_database(default_mode)
        self._configure_redis(default_mode)
        self._configure_celery(default_mode)
        self._configure_messaging(default_mode)
        self._configure_storage(default_mode)
        self._configure_ai_models(default_mode)
    
    def _configure_database(self, mode: ServiceMode):
        """Configure database service"""
        if mode == ServiceMode.MOCK:
            config = {
                'type': 'mock',
                'url': 'sqlite:///:memory:',
                'mock_data': True
            }
            available = True
        else:
            db_url = os.getenv('DATABASE_URL', 'postgresql://vigia_user:vigia_password@localhost:5432/vigia_dev')
            config = {
                'type': 'postgresql',
                'url': db_url,
                'pool_size': 10,
                'max_overflow': 20
            }
            available = self._check_database_connection(db_url)
        
        self.configs[ServiceType.DATABASE] = ServiceConfig(
            service_type=ServiceType.DATABASE,
            mode=mode,
            config=config,
            available=available,
            health_check_url=f"{config.get('url', '')}/health" if available else None
        )
    
    def _configure_redis(self, mode: ServiceMode):
        """Configure Redis cache service"""
        if mode == ServiceMode.MOCK:
            config = {
                'type': 'mock',
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'mock_data': True
            }
            available = True
        else:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            config = {
                'type': 'redis',
                'url': redis_url,
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'db': int(os.getenv('REDIS_DB', 0)),
                'password': os.getenv('REDIS_PASSWORD'),
                'max_connections': 20
            }
            available = self._check_redis_connection(config)
        
        self.configs[ServiceType.CACHE] = ServiceConfig(
            service_type=ServiceType.CACHE,
            mode=mode,
            config=config,
            available=available
        )
    
    def _configure_celery(self, mode: ServiceMode):
        """Configure Celery task queue"""
        if mode == ServiceMode.MOCK or not self._is_celery_available():
            config = {
                'type': 'mock',
                'broker_url': 'memory://',
                'result_backend': 'cache+memory://',
                'always_eager': True,
                'mock_tasks': True
            }
            available = True
            actual_mode = ServiceMode.MOCK
        else:
            redis_config = self.configs.get(ServiceType.CACHE, {}).config
            broker_url = redis_config.get('url', 'redis://localhost:6379/0')
            config = {
                'type': 'celery',
                'broker_url': broker_url,
                'result_backend': broker_url,
                'always_eager': False,
                'task_serializer': 'json',
                'accept_content': ['json'],
                'result_serializer': 'json'
            }
            available = self.configs.get(ServiceType.CACHE, ServiceConfig(ServiceType.CACHE, mode, {}, False)).available
            actual_mode = mode
        
        self.configs[ServiceType.CELERY] = ServiceConfig(
            service_type=ServiceType.CELERY,
            mode=actual_mode,
            config=config,
            available=available
        )
    
    def _configure_messaging(self, mode: ServiceMode):
        """Configure messaging services (WhatsApp, Slack)"""
        if mode == ServiceMode.MOCK:
            config = {
                'type': 'mock',
                'whatsapp': {'mock': True, 'webhook_url': 'http://localhost:8001/mock'},
                'slack': {'mock': True, 'webhook_url': 'http://localhost:8002/mock'},
                'sendgrid': {'mock': True, 'api_key': 'mock_key'}
            }
            available = True
        else:
            config = {
                'type': 'real',
                'whatsapp': {
                    'account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
                    'auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
                    'from_number': os.getenv('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886')
                },
                'slack': {
                    'bot_token': os.getenv('SLACK_BOT_TOKEN'),
                    'signing_secret': os.getenv('SLACK_SIGNING_SECRET'),
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL')
                },
                'sendgrid': {
                    'api_key': os.getenv('SENDGRID_API_KEY'),
                    'from_email': os.getenv('SENDGRID_FROM_EMAIL', 'noreply@vigia.medical')
                }
            }
            available = any([
                config['whatsapp']['account_sid'],
                config['slack']['bot_token'],
                config['sendgrid']['api_key']
            ])
        
        self.configs[ServiceType.MESSAGING] = ServiceConfig(
            service_type=ServiceType.MESSAGING,
            mode=mode,
            config=config,
            available=available
        )
    
    def _configure_storage(self, mode: ServiceMode):
        """Configure storage services (Supabase)"""
        if mode == ServiceMode.MOCK:
            config = {
                'type': 'mock',
                'supabase': {'mock': True, 'url': 'http://localhost:8003/mock'},
                'local_storage': {'path': './data/mock_storage'}
            }
            available = True
        else:
            config = {
                'type': 'real',
                'supabase': {
                    'url': os.getenv('SUPABASE_URL'),
                    'key': os.getenv('SUPABASE_KEY'),
                    'service_role_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY')
                },
                'local_storage': {'path': os.getenv('UPLOAD_FOLDER', './data/uploads')}
            }
            available = bool(config['supabase']['url'] and config['supabase']['key'])
        
        self.configs[ServiceType.STORAGE] = ServiceConfig(
            service_type=ServiceType.STORAGE,
            mode=mode,
            config=config,
            available=available
        )
    
    def _configure_ai_models(self, mode: ServiceMode):
        """Configure AI model services with MONAI + YOLOv5 adaptive architecture"""
        if mode == ServiceMode.MOCK:
            config = {
                'type': 'mock',
                'detection_strategy': 'mock',
                'yolo': {'mock': True, 'model_path': 'mock_yolo'},
                'monai': {'mock': True, 'model_path': 'mock_monai'},
                'medgemma': {'mock': True, 'endpoint': 'http://localhost:8004/mock'},
                'embeddings': {'mock': True, 'model': 'mock_embeddings'}
            }
            available = True
        else:
            use_local_ai = os.getenv('USE_LOCAL_AI', 'true').lower() == 'true'
            detection_strategy = os.getenv('DETECTION_STRATEGY', 'monai_primary').lower()
            
            config = {
                'type': 'real',
                'detection_strategy': detection_strategy,  # monai_primary, yolo_primary, adaptive
                'adaptive_detection': {
                    'primary_engine': os.getenv('PRIMARY_DETECTION_ENGINE', 'monai'),
                    'backup_engine': os.getenv('BACKUP_DETECTION_ENGINE', 'yolo'),
                    'monai_timeout': float(os.getenv('MONAI_TIMEOUT', 8.0)),
                    'intelligent_fallback': os.getenv('INTELLIGENT_FALLBACK', 'true').lower() == 'true',
                    'confidence_threshold_monai': float(os.getenv('MONAI_CONFIDENCE_THRESHOLD', 0.7)),
                    'confidence_threshold_yolo': float(os.getenv('YOLO_CONFIDENCE_THRESHOLD', 0.6))
                },
                'monai': {
                    'model_path': os.getenv('MONAI_MODEL_PATH', './models/monai_lpp_model.pt'),
                    'medical_grade': True,
                    'preprocessing_pipeline': 'medical_standard',
                    'device': 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu',
                    'precision_target': '90-95%',
                    'medical_compliance': ['NPUAP', 'EPUAP', 'PPPIA']
                },
                'yolo': {
                    'model_path': os.getenv('YOLO_MODEL_PATH', './models/vigia_lpp_yolo.pt'),
                    'confidence_threshold': float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', 0.25)),
                    'medical_grade': False,
                    'precision_target': '85-90%',
                    'backup_role': True
                },
                'medgemma': {
                    'use_local': use_local_ai,
                    'ollama_model': os.getenv('MEDGEMMA_MODEL_PATH', 'medgemma:7b'),
                    'vertex_ai_model': 'medgemma-7b-it',
                    'project_id': os.getenv('GOOGLE_CLOUD_PROJECT'),
                    'region': os.getenv('VERTEX_AI_REGION', 'us-central1')
                },
                'embeddings': {
                    'model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'device': 'cpu'
                }
            }
            available = self._check_ai_models_available(config)
        
        self.configs[ServiceType.AI_MODEL] = ServiceConfig(
            service_type=ServiceType.AI_MODEL,
            mode=mode,
            config=config,
            available=available
        )
    
    def _check_database_connection(self, db_url: str) -> bool:
        """Check if database is available"""
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return False
    
    def _check_redis_connection(self, config: Dict[str, Any]) -> bool:
        """Check if Redis is available"""
        try:
            import redis
            r = redis.Redis(
                host=config['host'],
                port=config['port'],
                db=config['db'],
                password=config.get('password')
            )
            r.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return False
    
    def _is_celery_available(self) -> bool:
        """Check if Celery is installed"""
        try:
            import celery
            return True
        except ImportError:
            return False
    
    def _check_ai_models_available(self, config: Dict[str, Any]) -> bool:
        """Check if AI models are available"""
        # For now, assume available if config is present
        # In production, this would check model files/endpoints
        return True
    
    def get_config(self, service_type: ServiceType) -> Optional[ServiceConfig]:
        """Get configuration for a service type"""
        return self.configs.get(service_type)
    
    def is_service_available(self, service_type: ServiceType) -> bool:
        """Check if a service is available"""
        config = self.get_config(service_type)
        return config.available if config else False
    
    def is_using_mocks(self, service_type: ServiceType) -> bool:
        """Check if a service is using mocks"""
        config = self.get_config(service_type)
        return config.mode == ServiceMode.MOCK if config else True
    
    def get_service_summary(self) -> Dict[str, Any]:
        """Get summary of all service configurations"""
        summary = {
            'environment': self.environment,
            'is_testing': self.is_testing,
            'services': {}
        }
        
        for service_type, config in self.configs.items():
            summary['services'][service_type.value] = {
                'mode': config.mode.value,
                'available': config.available,
                'type': config.config.get('type', 'unknown')
            }
        
        return summary
    
    def log_configuration(self):
        """Log current service configuration"""
        summary = self.get_service_summary()
        
        logger.info("=== Vigia Service Configuration ===")
        logger.info(f"Environment: {summary['environment']}")
        logger.info(f"Testing Mode: {summary['is_testing']}")
        logger.info("Services:")
        
        for service_name, service_info in summary['services'].items():
            status = "✅" if service_info['available'] else "❌"
            mode_icon = "🧪" if service_info['mode'] == 'mock' else "🔧"
            logger.info(f"  {status} {mode_icon} {service_name}: {service_info['mode']} ({service_info['type']})")


# Global service configuration manager
_service_manager: Optional[ServiceConfigManager] = None


def get_service_config(force_mode: Optional[ServiceMode] = None) -> ServiceConfigManager:
    """Get or create global service configuration manager"""
    global _service_manager
    
    if _service_manager is None or force_mode:
        _service_manager = ServiceConfigManager(force_mode)
        _service_manager.log_configuration()
    
    return _service_manager


def is_service_available(service_type: ServiceType) -> bool:
    """Check if a service is available"""
    return get_service_config().is_service_available(service_type)


def is_using_mocks(service_type: ServiceType) -> bool:
    """Check if a service is using mocks"""
    return get_service_config().is_using_mocks(service_type)


def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    config = get_service_config().get_config(ServiceType.DATABASE)
    return config.config if config else {}


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration"""
    config = get_service_config().get_config(ServiceType.CACHE)
    return config.config if config else {}


def get_celery_config() -> Dict[str, Any]:
    """Get Celery configuration"""
    config = get_service_config().get_config(ServiceType.CELERY)
    return config.config if config else {}


def get_ai_model_config() -> Dict[str, Any]:
    """Get AI model configuration with MONAI support"""
    config = get_service_config().get_config(ServiceType.AI_MODEL)
    return config.config if config else {}


def get_detection_strategy() -> str:
    """Get current detection strategy (monai_primary, yolo_primary, adaptive)"""
    config = get_ai_model_config()
    return config.get('detection_strategy', 'monai_primary')


def get_adaptive_detection_config() -> Dict[str, Any]:
    """Get adaptive detection configuration"""
    config = get_ai_model_config()
    return config.get('adaptive_detection', {
        'primary_engine': 'monai',
        'backup_engine': 'yolo',
        'monai_timeout': 8.0,
        'intelligent_fallback': True,
        'confidence_threshold_monai': 0.7,
        'confidence_threshold_yolo': 0.6
    })


def is_monai_primary() -> bool:
    """Check if MONAI is configured as primary detection engine"""
    strategy = get_detection_strategy()
    adaptive_config = get_adaptive_detection_config()
    return (strategy == 'monai_primary' or 
            (strategy == 'adaptive' and adaptive_config.get('primary_engine') == 'monai'))


def should_use_medical_grade_detection() -> bool:
    """Determine if medical-grade detection (MONAI) should be used"""
    return is_monai_primary() and not is_using_mocks(ServiceType.AI_MODEL)


if __name__ == "__main__":
    # Test service configuration
    import sys
    
    mode = None
    if len(sys.argv) > 1:
        try:
            mode = ServiceMode(sys.argv[1])
        except ValueError:
            print(f"Invalid mode: {sys.argv[1]}")
            print(f"Valid modes: {[m.value for m in ServiceMode]}")
            sys.exit(1)
    
    manager = get_service_config(mode)
    summary = manager.get_service_summary()
    
    print("\n" + "="*50)
    print("VIGIA SERVICE CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Environment: {summary['environment']}")
    print(f"Testing Mode: {summary['is_testing']}")
    print("\nServices:")
    
    for service_name, service_info in summary['services'].items():
        status = "✅ Available" if service_info['available'] else "❌ Unavailable"
        mode_desc = "🧪 Mock" if service_info['mode'] == 'mock' else "🔧 Real"
        print(f"  {service_name.upper():<12} {status:<15} {mode_desc} ({service_info['type']})")
    
    print("\n" + "="*50)