"""
Protocol Indexer - Mock implementation for compatibility
=======================================================
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ProtocolIndexer:
    """Mock protocol indexer for compatibility"""
    
    def __init__(self):
        logger.info("ProtocolIndexer initialized (mock mode)")
    
    def index_protocol(self, protocol_data: Dict[str, Any]) -> str:
        """Index medical protocol"""
        logger.info("Indexing protocol (mock)")
        return "mock_protocol_id"
    
    def search_protocols(self, query: str) -> List[Dict[str, Any]]:
        """Search medical protocols"""
        logger.info(f"Searching protocols: {query[:50]}... (mock)")
        return []

__all__ = ['ProtocolIndexer']