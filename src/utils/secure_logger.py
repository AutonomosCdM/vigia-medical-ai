"""
Secure logging utilities for Vigia
Ensures sensitive data is properly masked in logs
"""

import logging
import re
import json
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import inspect
from pathlib import Path

from vigia_detect.utils.security_validator import security_validator


class SecureFormatter(logging.Formatter):
    """Custom formatter that masks sensitive data"""
    
    # Patterns to mask in logs
    SENSITIVE_PATTERNS = [
        # API Keys and tokens
        (r'(api[_-]?key|token|secret|password|auth|bearer)\s*[=:]\s*["\']?([^"\'\s,}]+)["\']?', 
         r'\1=***MASKED***'),
        # Patient codes (keep format but mask numbers)
        (r'([A-Z]{2})-(\d{4})-(\d{3})', r'\1-XXXX-XXX'),
        # Phone numbers
        (r'\+?\d{10,15}', lambda m: f'+***{m.group()[-4:]}'),
        # Email addresses
        (r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 
         r'***@\2'),
        # URLs with credentials
        (r'(https?://)([^:]+):([^@]+)@', r'\1***:***@'),
        # File paths with usernames
        (r'/home/([^/]+)/', r'/home/***/'),
        (r'/Users/([^/]+)/', r'/Users/***/'),
        # IP addresses (partial masking)
        (r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b', 
         r'\1.\2.***.***'),
    ]
    
    # Fields to completely redact
    REDACT_FIELDS = [
        'password', 'secret', 'token', 'api_key', 'auth_token',
        'private_key', 'ssh_key', 'credit_card', 'ssn',
        'medical_record_number', 'patient_name', 'patient_details'
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive data masking"""
        # First, apply standard formatting
        msg = super().format(record)
        
        # Mask sensitive patterns
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            if callable(replacement):
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            else:
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
        
        # Handle structured data in extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key.lower() in self.REDACT_FIELDS:
                    setattr(record, key, '***REDACTED***')
                elif isinstance(value, (dict, list)):
                    setattr(record, key, self._mask_structured_data(value))
        
        return msg
    
    def _mask_structured_data(self, data: Union[Dict, List]) -> Union[Dict, List]:
        """Recursively mask sensitive data in structured formats"""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if key.lower() in self.REDACT_FIELDS:
                    masked[key] = '***REDACTED***'
                elif isinstance(value, (dict, list)):
                    masked[key] = self._mask_structured_data(value)
                elif isinstance(value, str):
                    # Apply pattern masking to string values
                    masked_value = value
                    for pattern, replacement in self.SENSITIVE_PATTERNS[:5]:  # Skip complex patterns
                        if not callable(replacement):
                            masked_value = re.sub(pattern, replacement, masked_value, flags=re.IGNORECASE)
                    masked[key] = masked_value
                else:
                    masked[key] = value
            return masked
        
        elif isinstance(data, list):
            return [self._mask_structured_data(item) if isinstance(item, (dict, list)) else item 
                   for item in data]
        
        return data


class SecureLogger:
    """Secure logger with automatic sensitive data masking"""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        """
        Initialize secure logger
        
        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with secure formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            SecureFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(
                SecureFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def _sanitize_args(self, args: tuple, kwargs: dict) -> tuple:
        """Sanitize function arguments before logging"""
        # Sanitize positional arguments
        sanitized_args = []
        for arg in args:
            if isinstance(arg, str) and any(field in arg.lower() for field in SecureFormatter.REDACT_FIELDS):
                sanitized_args.append('***REDACTED***')
            else:
                sanitized_args.append(arg)
        
        # Sanitize keyword arguments
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if key.lower() in SecureFormatter.REDACT_FIELDS:
                sanitized_kwargs[key] = '***REDACTED***'
            elif isinstance(value, str) and 'password' in value.lower():
                sanitized_kwargs[key] = '***REDACTED***'
            else:
                sanitized_kwargs[key] = value
        
        return tuple(sanitized_args), sanitized_kwargs
    
    def info(self, msg: str, *args, **kwargs):
        """Log info with automatic masking"""
        sanitized_args, sanitized_kwargs = self._sanitize_args(args, kwargs)
        self.logger.info(msg, *sanitized_args, extra=sanitized_kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning with automatic masking"""
        sanitized_args, sanitized_kwargs = self._sanitize_args(args, kwargs)
        self.logger.warning(msg, *sanitized_args, extra=sanitized_kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error with automatic masking"""
        sanitized_args, sanitized_kwargs = self._sanitize_args(args, kwargs)
        self.logger.error(msg, *sanitized_args, extra=sanitized_kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug with automatic masking"""
        sanitized_args, sanitized_kwargs = self._sanitize_args(args, kwargs)
        self.logger.debug(msg, *sanitized_args, extra=sanitized_kwargs)
    
    def log_api_request(self, method: str, url: str, headers: Dict[str, str], 
                       body: Optional[Any] = None):
        """Safely log API request"""
        # Mask sensitive headers
        safe_headers = {}
        for key, value in headers.items():
            if key.lower() in ['authorization', 'x-api-key', 'x-auth-token']:
                safe_headers[key] = '***MASKED***'
            else:
                safe_headers[key] = value
        
        # Mask sensitive body fields
        safe_body = None
        if body:
            if isinstance(body, dict):
                safe_body = SecureFormatter()._mask_structured_data(body)
            else:
                safe_body = str(body)[:100] + '...' if len(str(body)) > 100 else str(body)
        
        self.info(f"API Request: {method} {url}", 
                 headers=safe_headers, 
                 body=safe_body)
    
    def log_patient_action(self, action: str, patient_code: str, 
                          details: Optional[Dict] = None):
        """Log patient-related action with privacy protection"""
        # Mask patient code
        masked_code = re.sub(r'([A-Z]{2})-(\d{4})-(\d{3})', r'\1-XXXX-XXX', patient_code)
        
        # Log with masked data
        self.info(f"Patient Action: {action}", 
                 patient_code=masked_code,
                 details=details or {})
    
    def audit(self, event: str, details: Optional[Dict] = None):
        """Log audit event for compliance and security tracking"""
        audit_entry = {
            'audit_event': event,
            'timestamp': self._get_timestamp(),
            'compliance_level': 'audit',
            'details': details or {}
        }
        
        # Use info level for audit events
        self.info(f"AUDIT: {event}", extra=audit_entry)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


def secure_log(action: str = "function_call"):
    """
    Decorator to automatically log function calls securely
    
    Usage:
        @secure_log("process_image")
        def process_image(image_path, patient_code):
            ...
    """
    def decorator(func):
        logger = SecureLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Log function call with masked sensitive params
            safe_args = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_name.lower() in SecureFormatter.REDACT_FIELDS:
                    safe_args[param_name] = '***REDACTED***'
                elif isinstance(param_value, str) and len(param_value) > 100:
                    safe_args[param_name] = param_value[:50] + '...TRUNCATED'
                else:
                    safe_args[param_name] = param_value
            
            logger.info(f"{action} started: {func.__name__}", **safe_args)
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"{action} completed: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"{action} failed: {func.__name__}", error=str(e))
                raise
        
        return wrapper
    return decorator


# Create a global secure logger instance
secure_logger = SecureLogger('vigia.security')


# Example usage functions
def get_secure_logger(name: str, log_file: Optional[Path] = None) -> SecureLogger:
    """Get a secure logger instance"""
    return SecureLogger(name, log_file)


def log_security_event(event_type: str, severity: str, details: Dict[str, Any]):
    """Log a security event"""
    logger = get_secure_logger('vigia.security.events')
    
    # Add metadata
    event = {
        'event_type': event_type,
        'severity': severity,
        'timestamp': str(Path.ctime(Path.cwd())),
        'details': details
    }
    
    # Log based on severity
    if severity.upper() == 'CRITICAL':
        logger.error(f"SECURITY EVENT: {event_type}", **event)
    elif severity.upper() == 'HIGH':
        logger.warning(f"SECURITY EVENT: {event_type}", **event)
    else:
        logger.info(f"SECURITY EVENT: {event_type}", **event)