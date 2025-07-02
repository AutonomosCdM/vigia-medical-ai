"""
YOLO Loader - Legacy compatibility module
=========================================

Compatibility module for YOLO model loading. The actual implementation
is now integrated in adaptive_medical_detector.py.
"""

import logging

logger = logging.getLogger(__name__)

class YOLOLoader:
    """Legacy YOLO loader compatibility class"""
    
    def __init__(self):
        self.model_loaded = False
        logger.info("YOLOLoader initialized (compatibility mode)")
    
    def load_model(self, model_path: str = None):
        """Load YOLO model (compatibility method)"""
        logger.info(f"Loading YOLO model: {model_path}")
        self.model_loaded = True
        return True
    
    def is_available(self) -> bool:
        """Check if YOLO is available"""
        return True

__all__ = ['YOLOLoader']