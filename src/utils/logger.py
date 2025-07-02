"""
Logger utilities for Vigia Medical System
========================================

Simple wrapper around secure_logger for backward compatibility.
"""
import logging
from .secure_logger import get_secure_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with secure formatting.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return get_secure_logger(name)


# For backward compatibility
configure_logger = get_secure_logger