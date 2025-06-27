"""
Medical Telemetry - Mock implementation for compatibility
========================================================
"""

import logging

logger = logging.getLogger(__name__)

class MedicalTelemetry:
    """Mock medical telemetry for compatibility"""
    
    def __init__(self):
        logger.info("MedicalTelemetry initialized (mock mode)")
    
    def record_medical_event(self, event_data):
        """Record medical event"""
        logger.info("Recording medical event (mock)")

__all__ = ['MedicalTelemetry']