"""
Computer Vision Pipeline Module
==============================

MONAI medical imaging with YOLOv5 backup detection.
Professional medical image analysis for pressure injury detection.
"""

from .medical_detector_factory import create_medical_detector
from .adaptive_medical_detector import AdaptiveMedicalDetector

__all__ = ['create_medical_detector', 'AdaptiveMedicalDetector']