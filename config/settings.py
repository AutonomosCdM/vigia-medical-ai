"""
VIGIA Medical AI - Configuration Settings
========================================

Centralized configuration management for medical system components
with AWS MCP patterns and HIPAA compliance.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class VigiaSettings(BaseSettings):
    """VIGIA Medical AI Configuration Settings."""
    
    # Core Application Settings
    app_name: str = Field(default="vigia-medical-ai", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="VIGIA_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Medical Mode Configuration
    medical_mode: str = Field(default="production", env="MEDICAL_MODE")
    phi_tokenization_enabled: bool = Field(default=True, env="PHI_TOKENIZATION_ENABLED")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    
    # Medical Audit Trail
    audit_enabled: bool = Field(default=True, env="AUDIT_ENABLED")
    audit_log_retention_days: int = Field(default=2555, env="AUDIT_LOG_RETENTION_DAYS")  # 7 years HIPAA requirement
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_deployment: bool = Field(default=False, env="AWS_DEPLOYMENT")
    lambda_deployment: bool = Field(default=False, env="LAMBDA_DEPLOYMENT")
    
    # Database Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    postgres_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Medical AI Configuration
    medgemma_model_path: str = Field(default="medgemma-27b", env="MEDGEMMA_MODEL_PATH")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    
    # Security Configuration
    secret_key: str = Field(default="medical-ai-secret-key", env="SECRET_KEY")
    phi_encryption_key: Optional[str] = Field(default=None, env="PHI_ENCRYPTION_KEY")
    
    # External Services
    hume_api_key: Optional[str] = Field(default=None, env="HUME_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    slack_bot_token: Optional[str] = Field(default=None, env="SLACK_BOT_TOKEN")
    
    # Medical Compliance
    hipaa_compliance_mode: bool = Field(default=True, env="HIPAA_COMPLIANCE_MODE")
    medical_audit_required: bool = Field(default=True, env="MEDICAL_AUDIT_REQUIRED")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout_seconds: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    
    # File Storage
    medical_storage_path: str = Field(default="./storage/medical", env="MEDICAL_STORAGE_PATH")
    temp_file_retention_hours: int = Field(default=24, env="TEMP_FILE_RETENTION_HOURS")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env
        
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_medical_mode(self) -> bool:
        """Check if running in medical mode."""
        return self.medical_mode.lower() == "production"
    
    @property
    def is_aws_deployment(self) -> bool:
        """Check if running in AWS deployment."""
        return self.aws_deployment or self.lambda_deployment
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.log_level,
            "format": self.log_format,
            "file_path": self.log_file_path,
            "audit_enabled": self.audit_enabled,
            "audit_retention_days": self.audit_log_retention_days
        }
    
    def get_medical_config(self) -> Dict[str, Any]:
        """Get medical configuration."""
        return {
            "medical_mode": self.medical_mode,
            "phi_tokenization_enabled": self.phi_tokenization_enabled,
            "hipaa_compliance_mode": self.hipaa_compliance_mode,
            "medical_audit_required": self.medical_audit_required,
            "audit_retention_days": self.audit_log_retention_days
        }
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration."""
        return {
            "region": self.aws_region,
            "deployment": self.aws_deployment,
            "lambda_deployment": self.lambda_deployment
        }


# Global settings instance
settings = VigiaSettings()


def get_settings() -> VigiaSettings:
    """Get global settings instance."""
    return settings


# AWS MCP Configuration for Lambda
def configure_for_lambda():
    """Configure settings for AWS Lambda deployment."""
    global settings
    settings.aws_deployment = True
    settings.lambda_deployment = True
    settings.environment = "production"
    settings.medical_mode = "production"
    settings.log_level = "INFO"
    

# Medical System Configuration
def configure_for_medical_production():
    """Configure settings for medical production environment."""
    global settings
    settings.environment = "production"
    settings.medical_mode = "production"
    settings.phi_tokenization_enabled = True
    settings.hipaa_compliance_mode = True
    settings.medical_audit_required = True
    settings.audit_enabled = True


# Development Configuration
def configure_for_development():
    """Configure settings for development environment."""
    global settings
    settings.environment = "development"
    settings.debug = True
    settings.log_level = "DEBUG"