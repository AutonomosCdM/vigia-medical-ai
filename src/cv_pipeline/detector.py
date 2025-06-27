"""
LPP Detector - Legacy compatibility module
==========================================

Compatibility module for existing imports. The actual implementation
is now in real_lpp_detector.py and adaptive_medical_detector.py.
"""

import logging

logger = logging.getLogger(__name__)

class LPPDetector:
    """Legacy LPP detector compatibility class"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model_loaded = False
        logger.info("LPPDetector initialized (compatibility mode)")
    
    def detect(self, image_path: str):
        """Detect LPP in image (compatibility method)"""
        logger.info(f"Detecting LPP in: {image_path}")
        # Mock detection result
        return {
            'detections': [],
            'confidence': 0.0,
            'message': 'Mock detection (compatibility mode)'
        }
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return True

__all__ = ['LPPDetector']