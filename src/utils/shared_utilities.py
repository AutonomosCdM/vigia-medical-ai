"""
Shared utilities for logging, validation, and error handling.
Eliminates code duplication across the Vigia project.
"""
import logging
import functools
import traceback
from typing import Any, Dict, Optional, Callable, Union, List
from datetime import datetime
from pathlib import Path
import json

from config.settings import settings


class VigiaLogger:
    """
    Centralized logging utility for the Vigia project.
    Provides consistent logging across all modules.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with consistent configuration"""
        if name not in cls._loggers:
            logger = logging.getLogger(f"vigia.{name}")
            
            if not logger.handlers:
                # Create formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
                
                # File handler if logs directory exists
                logs_dir = Path("logs")
                if logs_dir.exists():
                    file_handler = logging.FileHandler(
                        logs_dir / f"{name}.log",
                        encoding='utf-8'
                    )
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                
                # Set level from settings
                logger.setLevel(getattr(logging, settings.log_level))
            
            cls._loggers[name] = logger
        
        return cls._loggers[name]


class VigiaValidator:
    """
    Centralized validation utilities for the Vigia project.
    Provides consistent validation across all modules.
    """
    
    @staticmethod
    def validate_patient_code(patient_code: str) -> Dict[str, Any]:
        """
        Validate patient code format.
        
        Args:
            patient_code: Patient identifier to validate
            
        Returns:
            Validation result
        """
        if not patient_code:
            return {"valid": False, "error": "Patient code is required"}
        
        # Expected format: XX-YYYY-NNN (e.g., CD-2025-001)
        parts = patient_code.split('-')
        if len(parts) != 3:
            return {
                "valid": False, 
                "error": "Patient code must follow format: XX-YYYY-NNN"
            }
        
        prefix, year, number = parts
        
        if len(prefix) != 2 or not prefix.isalpha():
            return {
                "valid": False,
                "error": "Patient code prefix must be 2 letters"
            }
        
        if len(year) != 4 or not year.isdigit():
            return {
                "valid": False,
                "error": "Patient code year must be 4 digits"
            }
        
        if len(number) != 3 or not number.isdigit():
            return {
                "valid": False,
                "error": "Patient code number must be 3 digits"
            }
        
        return {"valid": True, "patient_code": patient_code}
    
    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Validation result
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"valid": False, "error": f"File does not exist: {file_path}"}
        
        if not path.is_file():
            return {"valid": False, "error": f"Path is not a file: {file_path}"}
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if path.suffix.lower() not in valid_extensions:
            return {
                "valid": False,
                "error": f"Invalid image format. Supported: {', '.join(valid_extensions)}"
            }
        
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if path.stat().st_size > max_size:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: 50MB"
            }
        
        return {"valid": True, "file_path": str(path)}
    
    @staticmethod
    def validate_detection_confidence(confidence: float) -> Dict[str, Any]:
        """
        Validate detection confidence threshold.
        
        Args:
            confidence: Confidence threshold (0-1)
            
        Returns:
            Validation result
        """
        if not isinstance(confidence, (int, float)):
            return {
                "valid": False,
                "error": "Confidence must be a number"
            }
        
        if not 0 <= confidence <= 1:
            return {
                "valid": False,
                "error": "Confidence must be between 0 and 1"
            }
        
        return {"valid": True, "confidence": float(confidence)}


class VigiaErrorHandler:
    """
    Centralized error handling utilities for the Vigia project.
    Provides consistent error handling and reporting.
    """
    
    @staticmethod
    def handle_exception(logger: logging.Logger, 
                        operation: str,
                        exception: Exception,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle and log exceptions consistently.
        
        Args:
            logger: Logger instance
            operation: Description of the operation that failed
            exception: The exception that occurred
            context: Additional context information
            
        Returns:
            Standardized error response
        """
        error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        error_details = {
            "error_id": error_id,
            "operation": operation,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        # Log the error
        logger.error(
            f"Operation '{operation}' failed (ID: {error_id}): "
            f"{type(exception).__name__}: {str(exception)}"
        )
        
        # Log stack trace for debugging
        logger.debug(f"Stack trace for error {error_id}:", exc_info=True)
        
        return {
            "success": False,
            "error": error_details,
            "user_message": f"Operation failed. Error ID: {error_id}"
        }
    
    @staticmethod
    def create_success_response(data: Any, 
                              operation: str,
                              message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create standardized success response.
        
        Args:
            data: Response data
            operation: Description of the operation
            message: Optional success message
            
        Returns:
            Standardized success response
        """
        return {
            "success": True,
            "operation": operation,
            "data": data,
            "message": message or f"Operation '{operation}' completed successfully",
            "timestamp": datetime.now().isoformat()
        }


def with_error_handling(operation_name: str):
    """
    Decorator for consistent error handling.
    
    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = VigiaLogger.get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"Operation '{operation_name}' completed successfully")
                return result
                
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                
                return VigiaErrorHandler.handle_exception(
                    logger, operation_name, e, context
                )
        
        return wrapper
    return decorator


def with_validation(validators: List[Callable]) -> Callable:
    """
    Decorator for input validation.
    
    Args:
        validators: List of validation functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run validators
            for validator in validators:
                try:
                    validation_result = validator(*args, **kwargs)
                    if not validation_result.get("valid", True):
                        return {
                            "success": False,
                            "error": validation_result.get("error", "Validation failed")
                        }
                except Exception as e:
                    logger = VigiaLogger.get_logger(func.__module__)
                    logger.error(f"Validation error: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Validation error: {str(e)}"
                    }
            
            # If all validations pass, execute the function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class PerformanceTracker:
    """
    Performance tracking utility for monitoring operation times.
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.logger = VigiaLogger.get_logger("performance")
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed operation: {self.operation_name} after {duration:.2f}s")
    
    def get_duration(self) -> Optional[float]:
        """Get operation duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None