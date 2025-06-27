"""
Medical Image Storage - Mock implementation for compatibility
============================================================
"""

import logging
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AnatomicalRegion(Enum):
    """Anatomical regions for medical images"""
    SACRO = "sacro"
    TALON = "talon"
    CADERA = "cadera"
    COXIS = "coxis"
    UNKNOWN = "unknown"

class ImageType(Enum):
    """Medical image types"""
    PRESSURE_INJURY = "pressure_injury"
    NORMAL_SKIN = "normal_skin"
    DIAGNOSTIC = "diagnostic"

class MedicalImageStorage:
    """Mock medical image storage for compatibility"""
    
    def __init__(self):
        logger.info("MedicalImageStorage initialized (mock mode)")
    
    def store_image(self, image_data: bytes, metadata: Dict[str, Any]) -> str:
        """Store medical image"""
        logger.info("Storing medical image (mock)")
        return "mock_image_id"
    
    def get_image(self, image_id: str) -> Dict[str, Any]:
        """Get medical image"""
        logger.info(f"Getting medical image: {image_id} (mock)")
        return {'id': image_id, 'data': b'', 'mock': True}

__all__ = ['MedicalImageStorage', 'AnatomicalRegion', 'ImageType']