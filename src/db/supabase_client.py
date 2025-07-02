"""
Supabase Client - Mock implementation for compatibility
======================================================
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Mock Supabase client for compatibility"""
    
    def __init__(self):
        self.connected = False
        logger.info("SupabaseClient initialized (mock mode)")
    
    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock insert operation"""
        logger.info(f"Mock insert to {table}: {len(data)} fields")
        return {'success': True, 'id': 'mock_id', 'data': data}
    
    def select(self, table: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Mock select operation"""
        logger.info(f"Mock select from {table}")
        return []
    
    def update(self, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock update operation"""
        logger.info(f"Mock update in {table}")
        return {'success': True, 'updated': 1}

__all__ = ['SupabaseClient']