"""
AgentOps Client - Mock implementation for compatibility
======================================================
"""

import logging

logger = logging.getLogger(__name__)

class AgentOpsClient:
    """Mock AgentOps client for compatibility"""
    
    def __init__(self):
        logger.info("AgentOpsClient initialized (mock mode)")
    
    def start_session(self):
        """Start monitoring session"""
        logger.info("Starting AgentOps session (mock)")
        return "mock_session_id"
    
    def end_session(self):
        """End monitoring session"""
        logger.info("Ending AgentOps session (mock)")

__all__ = ['AgentOpsClient']