"""
Medical Detector Factory with Adaptive Architecture
=================================================

Factory for creating medical detectors with intelligent engine selection.
Supports MONAI primary + YOLOv5 backup architecture with configuration-driven selection.

Features:
- Configuration-driven detector creation
- MONAI medical-grade detection with YOLOv5 backup
- Backward compatibility with existing Vigia interfaces
- Enhanced audit trail and performance metrics
- HIPAA-compliant Batman tokenization support

Usage:
    factory = MedicalDetectorFactory()
    detector = factory.create_detector()
    result = await detector.detect_medical_condition(image_path, token_id)
"""

import logging
from typing import Optional, Dict, Any, Union
from enum import Enum

# Vigia configuration and existing detectors
from ..core.service_config import (
    get_ai_model_config, 
    get_detection_strategy,
    get_adaptive_detection_config,
    should_use_medical_grade_detection,
    is_using_mocks,
    ServiceType
)
from .adaptive_medical_detector import AdaptiveMedicalDetector, DetectionEngine
from .real_lpp_detector import PressureUlcerDetector

logger = logging.getLogger(__name__)


class DetectorType(Enum):
    """Available detector types"""
    ADAPTIVE_MEDICAL = "adaptive_medical"
    MONAI_PRIMARY = "monai_primary"
    YOLO_ONLY = "yolo_only"
    LEGACY_COMPATIBLE = "legacy_compatible"
    MOCK = "mock"


class MedicalDetectorFactory:
    """
    Factory for creating medical detectors with intelligent configuration.
    
    Automatically selects optimal detector based on:
    - Service configuration (MONAI availability, strategy)
    - Environment settings (mock vs real)
    - Backward compatibility requirements
    """
    
    def __init__(self, force_detector_type: Optional[DetectorType] = None):
        """
        Initialize detector factory.
        
        Args:
            force_detector_type: Force specific detector type (for testing)
        """
        self.force_detector_type = force_detector_type
        self.ai_config = get_ai_model_config()
        self.detection_strategy = get_detection_strategy()
        self.adaptive_config = get_adaptive_detection_config()
        
        logger.info(f"Medical Detector Factory initialized - Strategy: {self.detection_strategy}")
    
    def create_detector(self, 
                       detector_type: Optional[DetectorType] = None,
                       **kwargs) -> Union[AdaptiveMedicalDetector, PressureUlcerDetector]:
        """
        Create optimal medical detector based on configuration.
        
        Args:
            detector_type: Override detector type
            **kwargs: Additional configuration for detector
            
        Returns:
            Configured medical detector
        """
        # Determine detector type
        if detector_type:
            selected_type = detector_type
        elif self.force_detector_type:
            selected_type = self.force_detector_type
        else:
            selected_type = self._determine_optimal_detector_type()
        
        logger.info(f"Creating detector type: {selected_type.value}")
        
        # Create detector based on type
        if selected_type == DetectorType.ADAPTIVE_MEDICAL:
            return self._create_adaptive_medical_detector(**kwargs)
        elif selected_type == DetectorType.MONAI_PRIMARY:
            return self._create_monai_primary_detector(**kwargs)
        elif selected_type == DetectorType.YOLO_ONLY:
            return self._create_yolo_only_detector(**kwargs)
        elif selected_type == DetectorType.LEGACY_COMPATIBLE:
            return self._create_legacy_compatible_detector(**kwargs)
        elif selected_type == DetectorType.MOCK:
            return self._create_mock_detector(**kwargs)
        else:
            logger.warning(f"Unknown detector type: {selected_type}, falling back to legacy")
            return self._create_legacy_compatible_detector(**kwargs)
    
    def _determine_optimal_detector_type(self) -> DetectorType:
        """
        Determine optimal detector type based on configuration and environment.
        
        Returns:
            Optimal DetectorType for current environment
        """
        # Mock mode for testing
        if is_using_mocks(ServiceType.AI_MODEL):
            return DetectorType.MOCK
        
        # Check detection strategy
        if self.detection_strategy == 'monai_primary':
            # MONAI primary with YOLOv5 backup
            return DetectorType.ADAPTIVE_MEDICAL
        elif self.detection_strategy == 'yolo_primary':
            # YOLOv5 only (legacy mode)
            return DetectorType.YOLO_ONLY
        elif self.detection_strategy == 'adaptive':
            # Full adaptive detection with intelligent routing
            return DetectorType.ADAPTIVE_MEDICAL
        else:
            # Default to legacy for backward compatibility
            logger.info("Unknown strategy, using legacy compatibility mode")
            return DetectorType.LEGACY_COMPATIBLE
    
    def _create_adaptive_medical_detector(self, **kwargs) -> AdaptiveMedicalDetector:
        """Create adaptive medical detector with MONAI + YOLOv5"""
        config = self.ai_config
        
        # Extract model paths
        monai_config = config.get('monai', {})
        yolo_config = config.get('yolo', {})
        adaptive_config = config.get('adaptive_detection', {})
        
        # Override with kwargs
        detector_config = {
            'monai_model_path': kwargs.get('monai_model_path', monai_config.get('model_path')),
            'yolo_model_path': kwargs.get('yolo_model_path', yolo_config.get('model_path')),
            'monai_timeout': kwargs.get('monai_timeout', adaptive_config.get('monai_timeout', 8.0)),
            'confidence_threshold_monai': kwargs.get(
                'confidence_threshold_monai', 
                adaptive_config.get('confidence_threshold_monai', 0.7)
            ),
            'confidence_threshold_yolo': kwargs.get(
                'confidence_threshold_yolo',
                adaptive_config.get('confidence_threshold_yolo', 0.6)
            )
        }
        
        logger.info("Creating adaptive medical detector with MONAI primary + YOLOv5 backup")
        logger.info(f"Configuration: {detector_config}")
        
        return AdaptiveMedicalDetector(**detector_config)
    
    def _create_monai_primary_detector(self, **kwargs) -> AdaptiveMedicalDetector:
        """Create MONAI-primary detector (no YOLOv5 backup)"""
        config = self.ai_config
        monai_config = config.get('monai', {})
        
        detector_config = {
            'monai_model_path': kwargs.get('monai_model_path', monai_config.get('model_path')),
            'yolo_model_path': None,  # No backup
            'monai_timeout': kwargs.get('monai_timeout', 8.0),
            'confidence_threshold_monai': kwargs.get('confidence_threshold_monai', 0.7),
            'confidence_threshold_yolo': 0.0  # Not used
        }
        
        logger.info("Creating MONAI-only detector (no backup)")
        return AdaptiveMedicalDetector(**detector_config)
    
    def _create_yolo_only_detector(self, **kwargs) -> PressureUlcerDetector:
        """Create YOLOv5-only detector (legacy mode)"""
        config = self.ai_config
        yolo_config = config.get('yolo', {})
        
        model_path = kwargs.get('model_path', yolo_config.get('model_path'))
        
        logger.info("Creating YOLOv5-only detector (legacy mode)")
        return PressureUlcerDetector(model_path=model_path)
    
    def _create_legacy_compatible_detector(self, **kwargs) -> PressureUlcerDetector:
        """Create legacy-compatible detector for backward compatibility"""
        config = self.ai_config
        yolo_config = config.get('yolo', {})
        
        model_path = kwargs.get('model_path', yolo_config.get('model_path'))
        
        logger.info("Creating legacy-compatible detector")
        return PressureUlcerDetector(model_path=model_path)
    
    def _create_mock_detector(self, **kwargs) -> AdaptiveMedicalDetector:
        """Create mock detector for testing"""
        detector_config = {
            'monai_model_path': None,
            'yolo_model_path': None,
            'monai_timeout': 1.0,  # Fast timeout for testing
            'confidence_threshold_monai': 0.5,
            'confidence_threshold_yolo': 0.5
        }
        
        logger.info("Creating mock detector for testing")
        return AdaptiveMedicalDetector(**detector_config)
    
    def get_detector_capabilities(self, detector_type: Optional[DetectorType] = None) -> Dict[str, Any]:
        """
        Get capabilities of specified detector type.
        
        Args:
            detector_type: Detector type to analyze (or auto-detect)
            
        Returns:
            Dictionary of detector capabilities
        """
        if not detector_type:
            detector_type = self._determine_optimal_detector_type()
        
        capabilities = {
            'detector_type': detector_type.value,
            'medical_grade': False,
            'backup_available': False,
            'adaptive_routing': False,
            'timeout_handling': False,
            'confidence_thresholds': {},
            'supported_models': [],
            'precision_target': 'unknown'
        }
        
        if detector_type == DetectorType.ADAPTIVE_MEDICAL:
            capabilities.update({
                'medical_grade': True,
                'backup_available': True,
                'adaptive_routing': True,
                'timeout_handling': True,
                'confidence_thresholds': {
                    'monai': self.adaptive_config.get('confidence_threshold_monai', 0.7),
                    'yolo': self.adaptive_config.get('confidence_threshold_yolo', 0.6)
                },
                'supported_models': ['MONAI', 'YOLOv5'],
                'precision_target': '90-95% (MONAI), 85-90% (YOLOv5 backup)'
            })
        elif detector_type == DetectorType.MONAI_PRIMARY:
            capabilities.update({
                'medical_grade': True,
                'backup_available': False,
                'adaptive_routing': False,
                'timeout_handling': True,
                'supported_models': ['MONAI'],
                'precision_target': '90-95%'
            })
        elif detector_type in [DetectorType.YOLO_ONLY, DetectorType.LEGACY_COMPATIBLE]:
            yolo_config = self.ai_config.get('yolo', {})
            capabilities.update({
                'medical_grade': False,
                'backup_available': False,
                'adaptive_routing': False,
                'timeout_handling': False,
                'confidence_thresholds': {
                    'yolo': yolo_config.get('confidence_threshold', 0.25)
                },
                'supported_models': ['YOLOv5'],
                'precision_target': '85-90%'
            })
        elif detector_type == DetectorType.MOCK:
            capabilities.update({
                'medical_grade': False,
                'backup_available': True,
                'adaptive_routing': True,
                'timeout_handling': True,
                'supported_models': ['Mock'],
                'precision_target': 'variable (testing)'
            })
        
        return capabilities
    
    def log_configuration_summary(self):
        """Log comprehensive configuration summary"""
        detector_type = self._determine_optimal_detector_type()
        capabilities = self.get_detector_capabilities(detector_type)
        
        logger.info("=== Medical Detector Configuration Summary ===")
        logger.info(f"Detection Strategy: {self.detection_strategy}")
        logger.info(f"Selected Detector: {detector_type.value}")
        logger.info(f"Medical Grade: {'✅' if capabilities['medical_grade'] else '❌'}")
        logger.info(f"Backup Available: {'✅' if capabilities['backup_available'] else '❌'}")
        logger.info(f"Adaptive Routing: {'✅' if capabilities['adaptive_routing'] else '❌'}")
        logger.info(f"Timeout Handling: {'✅' if capabilities['timeout_handling'] else '❌'}")
        logger.info(f"Supported Models: {', '.join(capabilities['supported_models'])}")
        logger.info(f"Precision Target: {capabilities['precision_target']}")
        
        if capabilities['confidence_thresholds']:
            logger.info("Confidence Thresholds:")
            for model, threshold in capabilities['confidence_thresholds'].items():
                logger.info(f"  {model.upper()}: {threshold}")


# Global factory instance
_detector_factory: Optional[MedicalDetectorFactory] = None


def get_medical_detector_factory(force_detector_type: Optional[DetectorType] = None) -> MedicalDetectorFactory:
    """Get or create global medical detector factory"""
    global _detector_factory
    
    if _detector_factory is None or force_detector_type:
        _detector_factory = MedicalDetectorFactory(force_detector_type)
        _detector_factory.log_configuration_summary()
    
    return _detector_factory


def create_medical_detector(**kwargs) -> Union[AdaptiveMedicalDetector, PressureUlcerDetector]:
    """
    Convenience function to create optimal medical detector.
    
    Args:
        **kwargs: Configuration for detector
        
    Returns:
        Configured medical detector
    """
    factory = get_medical_detector_factory()
    return factory.create_detector(**kwargs)


def create_monai_primary_detector(**kwargs) -> AdaptiveMedicalDetector:
    """
    Create MONAI primary detector with YOLOv5 backup.
    
    Args:
        **kwargs: Configuration for detector
        
    Returns:
        AdaptiveMedicalDetector with MONAI primary
    """
    factory = get_medical_detector_factory()
    return factory.create_detector(detector_type=DetectorType.ADAPTIVE_MEDICAL, **kwargs)


def create_legacy_detector(**kwargs) -> PressureUlcerDetector:
    """
    Create legacy YOLOv5 detector for backward compatibility.
    
    Args:
        **kwargs: Configuration for detector
        
    Returns:
        PressureUlcerDetector (YOLOv5)
    """
    factory = get_medical_detector_factory()
    return factory.create_detector(detector_type=DetectorType.LEGACY_COMPATIBLE, **kwargs)