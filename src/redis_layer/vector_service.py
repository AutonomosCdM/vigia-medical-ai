"""
Vector Service - Mock implementation for compatibility
=====================================================
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorService:
    """Mock vector service for compatibility"""
    
    def __init__(self):
        logger.info("VectorService initialized (mock mode)")
    
    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        logger.info(f"Vector search: top_k={top_k} (mock)")
        return []
    
    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> str:
        """Add vector to index"""
        logger.info("Adding vector (mock)")
        return "mock_vector_id"

__all__ = ['VectorService']