"""
VIGIA Medical AI Settings Configuration
======================================

Centralized settings management with environment variable support
and medical-grade security defaults.
"""

import os
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum


class EnvironmentType(Enum):
    """Environment types for medical AI deployment."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"
    HIPAA_PRODUCTION = "hipaa_production"


@dataclass
class HIPAASettings:
    """HIPAA-specific configuration settings."""
    PHI_TOKENIZATION_ENABLED: bool = True
    AUDIT_LOGGING_ENABLED: bool = True
    ENCRYPTION_AT_REST: bool = True
    ACCESS_LOGGING_ENABLED: bool = True
    SESSION_TIMEOUT_MINUTES: int = 15
    MAX_FAILED_ATTEMPTS: int = 3
    PASSWORD_COMPLEXITY_REQUIRED: bool = True
    DATA_RETENTION_DAYS: int = 2555  # 7 years HIPAA requirement


@dataclass
class Settings:
    """Medical AI system settings with HIPAA compliance defaults."""
    
    # Medical AI Configuration
    REDIS_URL: str = "redis://localhost:6379/1"
    POSTGRES_URL: str = "postgresql://localhost:5432/vigia_medical"
    
    # Medical Processing Settings
    IMAGE_UPLOAD_MAX_SIZE: int = 50 * 1024 * 1024  # 50MB for medical images
    MEDICAL_SESSION_TIMEOUT: int = 900  # 15 minutes for HIPAA compliance
    
    # AI Model Configuration
    MEDGEMMA_MODEL_PATH: str = "models/medgemma-27b"
    MONAI_MODEL_PATH: str = "models/monai-medical"
    YOLOV5_MODEL_PATH: str = "models/yolov5-medical"
    
    # Google Cloud ADK Settings
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # Medical Communication
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_WHATSAPP_FROM: str = "whatsapp:+14155238886"
    
    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_SIGNING_SECRET: Optional[str] = None
    SLACK_MEDICAL_CHANNEL: str = "#medical-alerts"
    
    # Security Settings
    SECRET_KEY: str = "vigia-medical-ai-development-key-change-in-production"
    ENCRYPTION_KEY: Optional[str] = None
    
    # Medical File Storage
    MEDICAL_IMAGES_PATH: str = "storage/medical_images"
    AUDIT_LOGS_PATH: str = "storage/audit_logs"
    
    # Development Settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Environment Configuration
    ENVIRONMENT: EnvironmentType = EnvironmentType.DEVELOPMENT
    
    # HIPAA Configuration
    hipaa_settings: HIPAASettings = field(default_factory=HIPAASettings)
    
    def __post_init__(self) -> None:
        """Load values from environment variables and validate HIPAA compliance."""
        # Load environment type
        env_type = os.getenv('VIGIA_ENVIRONMENT', 'development')
        try:
            self.ENVIRONMENT = EnvironmentType(env_type)
        except ValueError:
            self.ENVIRONMENT = EnvironmentType.DEVELOPMENT
        
        # Load environment variables
        self._load_from_environment()
        
        # Apply environment-specific settings
        self._apply_environment_settings()
        
        # Validate HIPAA compliance for production
        if self.ENVIRONMENT in [EnvironmentType.PRODUCTION, EnvironmentType.HIPAA_PRODUCTION]:
            self._validate_hipaa_compliance()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        for field_name in self.__dataclass_fields__:
            if field_name in ['ENVIRONMENT', 'hipaa_settings']:
                continue
                
            env_value = os.getenv(f'VIGIA_{field_name}') or os.getenv(field_name)
            if env_value is not None:
                field_type = self.__dataclass_fields__[field_name].type
                
                # Handle Optional types
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                    field_type = field_type.__args__[0]
                
                # Convert based on type
                if field_type is bool:
                    setattr(self, field_name, env_value.lower() in ('true', '1', 'yes', 'on'))
                elif field_type is int:
                    setattr(self, field_name, int(env_value))
                elif field_type is float:
                    setattr(self, field_name, float(env_value))
                else:
                    setattr(self, field_name, env_value)
    
    def _apply_environment_settings(self) -> None:
        """Apply environment-specific configuration overrides."""
        if self.ENVIRONMENT == EnvironmentType.PRODUCTION:
            self.DEBUG = False
            self.LOG_LEVEL = "WARNING"
            self.hipaa_settings.PHI_TOKENIZATION_ENABLED = True
            self.hipaa_settings.AUDIT_LOGGING_ENABLED = True
            
        elif self.ENVIRONMENT == EnvironmentType.HIPAA_PRODUCTION:
            self.DEBUG = False
            self.LOG_LEVEL = "INFO"
            self.hipaa_settings.PHI_TOKENIZATION_ENABLED = True
            self.hipaa_settings.AUDIT_LOGGING_ENABLED = True
            self.hipaa_settings.ENCRYPTION_AT_REST = True
            
        elif self.ENVIRONMENT == EnvironmentType.TESTING:
            self.DEBUG = True
            self.LOG_LEVEL = "DEBUG"
            self.hipaa_settings.SESSION_TIMEOUT_MINUTES = 5  # Shorter for testing
    
    def _validate_hipaa_compliance(self) -> None:
        """Validate HIPAA compliance requirements for production."""
        required_secrets = [
            'SECRET_KEY', 'ENCRYPTION_KEY', 'TWILIO_AUTH_TOKEN', 
            'SLACK_BOT_TOKEN', 'GOOGLE_APPLICATION_CREDENTIALS'
        ]
        
        missing_secrets = []
        for secret in required_secrets:
            if not getattr(self, secret, None):
                missing_secrets.append(secret)
        
        if missing_secrets:
            raise ValueError(
                f"HIPAA Production requires these secrets: {', '.join(missing_secrets)}"
            )
        
        # Validate HIPAA settings
        if not self.hipaa_settings.PHI_TOKENIZATION_ENABLED:
            raise ValueError("PHI tokenization is required for HIPAA compliance")
        
        if not self.hipaa_settings.AUDIT_LOGGING_ENABLED:
            raise ValueError("Audit logging is required for HIPAA compliance")
    
    def get_database_url(self) -> str:
        """Get database URL with environment-specific defaults."""
        if self.ENVIRONMENT == EnvironmentType.TESTING:
            return "postgresql://localhost:5432/vigia_test"
        return self.POSTGRES_URL
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT in [EnvironmentType.PRODUCTION, EnvironmentType.HIPAA_PRODUCTION]

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings

# Medical environment validation
def validate_medical_environment() -> Dict[str, bool]:
    """Validate that critical medical environment variables are set."""
    validations = {
        "redis_available": bool(settings.REDIS_URL),
        "medical_storage_configured": bool(settings.MEDICAL_IMAGES_PATH),
        "hipaa_audit_enabled": settings.HIPAA_AUDIT_ENABLED,
        "phi_tokenization_enabled": settings.PHI_TOKENIZATION_ENABLED,
        "secret_key_configured": settings.SECRET_KEY != "vigia-medical-ai-development-key-change-in-production"
    }
    
    return validations

def get_medical_paths() -> Dict[str, Path]:
    """Get important medical system paths."""
    base_path = Path.cwd()
    
    return {
        "medical_images": base_path / settings.MEDICAL_IMAGES_PATH,
        "audit_logs": base_path / settings.AUDIT_LOGS_PATH,
        "models": base_path / "models",
        "config": base_path / "config",
        "src": base_path / "src"
    }