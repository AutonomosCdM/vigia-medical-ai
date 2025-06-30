"""
Medical Telemetry - Mock implementation for compatibility
========================================================
"""

import logging

logger = logging.getLogger(__name__)

class MedicalTelemetry:
    """Mock medical telemetry for compatibility"""
    
    def __init__(self, app_id=None, environment=None, enable_phi_protection=None, **kwargs):
        self.app_id = app_id
        self.environment = environment
        self.enable_phi_protection = enable_phi_protection
        logger.info(f"MedicalTelemetry initialized (mock mode) - App: {app_id}, Env: {environment}")
    
    def record_medical_event(self, event_data):
        """Record medical event"""
        logger.info("Recording medical event (mock)")

__all__ = ['MedicalTelemetry']