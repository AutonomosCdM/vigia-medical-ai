"""
VIGIA Medical AI - Training Database Mock
=========================================

Mock/simplified version of training database for testing dataset integration
without aioredis dependency issues.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class NPUAPGrade(Enum):
    """NPUAP Pressure Injury Classification"""
    STAGE_0 = "0"  # No visible pressure injury
    STAGE_1 = "1"  # Non-blanchable erythema
    STAGE_2 = "2"  # Partial thickness skin loss
    STAGE_3 = "3"  # Full thickness skin loss
    STAGE_4 = "4"  # Full thickness skin and tissue loss
    UNSTAGEABLE = "U"  # Unstageable/unclassified
    DEEP_TISSUE = "DTI"  # Deep tissue pressure injury

class TrainingDatabase:
    """Mock training database for testing"""
    
    def __init__(self, storage_path: str = "./data/training"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._mock_mode = True
        
        logger.info(f"TrainingDatabase (mock) initialized - Storage: {self.storage_path}")
    
    async def initialize(self) -> None:
        """Initialize mock database"""
        self._initialized = True
        logger.info("TrainingDatabase (mock) initialized successfully")
    
    async def store_medical_image(self,
                                image_path: str,
                                npuap_grade: NPUAPGrade,
                                batman_token: str,
                                dataset_source: str = None,
                                split_type: str = "train",
                                integration_id: str = None,
                                medical_metadata: Dict[str, Any] = None,
                                patient_context: Dict[str, Any] = None) -> str:
        """Store medical image metadata (mock implementation)"""
        
        # Mock storage - just log the operation
        logger.info(f"Mock storage: {image_path} - Grade: {npuap_grade.value}")
        
        # Return mock record ID
        return f"mock_record_{integration_id}_{len(image_path)}"
    
    async def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics (mock)"""
        return {
            "total_images": 0,
            "npuap_distribution": {},
            "last_updated": datetime.now().isoformat(),
            "mock_mode": True
        }