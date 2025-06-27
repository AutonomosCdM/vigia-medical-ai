"""
ADK Wrapper - Mock implementation for compatibility
===================================================
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)

def adk_agent_wrapper(func):
    """Mock ADK agent wrapper decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"ADK wrapper: {func.__name__} (mock)")
        return func(*args, **kwargs)
    return wrapper

__all__ = ['adk_agent_wrapper']