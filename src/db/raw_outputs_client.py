"""
Raw Outputs Client - Mock implementation for compatibility
=========================================================
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RawOutputsClient:
    """Mock raw outputs client for compatibility"""
    
    def __init__(self):
        logger.info("RawOutputsClient initialized (mock mode)")
    
    def store_raw_output(self, output_data: Dict[str, Any]) -> str:
        """Store raw output data"""
        logger.info("Storing raw output (mock)")
        return "mock_output_id"
    
    def get_raw_output(self, output_id: str) -> Optional[Dict[str, Any]]:
        """Get raw output by ID"""
        logger.info(f"Getting raw output: {output_id} (mock)")
        return {'id': output_id, 'data': {}, 'mock': True}

__all__ = ['RawOutputsClient']