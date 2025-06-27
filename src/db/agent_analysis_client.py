"""
Agent Analysis Client - Mock implementation for compatibility
============================================================
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AgentAnalysisClient:
    """Mock agent analysis client for compatibility"""
    
    def __init__(self):
        logger.info("AgentAnalysisClient initialized (mock mode)")
    
    def store_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Store agent analysis"""
        logger.info("Storing analysis (mock)")
        return "mock_analysis_id"

__all__ = ['AgentAnalysisClient']