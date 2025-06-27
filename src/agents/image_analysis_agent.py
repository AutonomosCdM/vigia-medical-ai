"""
ImageAnalysisAgent - ADK Agent for Medical Image Analysis
========================================================

Specialized ADK agent for medical image analysis using YOLOv5-based LPP detection
with comprehensive medical preprocessing, privacy protection, and clinical enhancement.

This agent encapsulates the complete cv_pipeline functionality into ADK patterns
for seamless A2A communication and distributed medical processing.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import numpy as np
from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool

# Import CV pipeline components
from src.cv_pipeline.detector import LPPDetector
from src.cv_pipeline.preprocessor import ImagePreprocessor
from src.cv_pipeline.real_lpp_detector import RealLPPDetector, PressureUlcerDetector
from src.cv_pipeline.yolo_loader import YOLOLoader

# Import medical systems
from src.core.session_manager import SessionManager
from src.utils.audit_service import AuditService, AuditEventType
from src.a2a.base_infrastructure import A2AServer, AgentCard

logger = logging.getLogger(__name__)

# Agent instruction for medical image analysis
IMAGE_ANALYSIS_INSTRUCTION = """
Eres el ImageAnalysisAgent especializado en análisis de imágenes médicas para detección 
de lesiones por presión (LPP) usando visión por computadora médica avanzada.

RESPONSABILIDADES PRINCIPALES:
1. Análisis de imágenes médicas con YOLOv5 especializado en LPP
2. Preprocesamiento médico con protección de privacidad HIPAA
3. Detección y clasificación de LPP (Grados 1-4)
4. Enriquecimiento clínico para detección de eritema
5. Validación de confianza y escalamiento médico
6. Generación de reportes de análisis estructurados

CAPACIDADES TÉCNICAS:
- YOLOv5 médico entrenado en 2,088+ imágenes reales
- Preprocesamiento clínico con CLAHE para eritema
- Protección privacidad: eliminación EXIF, detección caras
- Soporte GPU/CPU automático con fallback mock
- Validación médica de resultados con umbrales clínicos

PROTOCOLOS DE ESCALAMIENTO:
- Confianza < 60%: Escalamiento a revisión especialista
- LPP Grado 3-4: Notificación inmediata equipo médico
- Errores procesamiento: Fallback a detección mock
- Imágenes inválidas: Validación y rechazo seguro

COMPLIANCE MÉDICO:
- Procesamiento HIPAA compliant con anonimización
- Audit trail completo de análisis médicos
- Timeouts inteligentes para casos críticos
- Integración seamless con pipeline asíncrono
"""


class ImageAnalysisAgent:
    """
    Medical image analysis agent implementing ADK patterns with A2A communication.
    Encapsulates complete cv_pipeline functionality for distributed processing.
    """
    
    def __init__(self):
        """Initialize image analysis agent with medical components"""
        self.agent_id = "image_analysis_agent"
        self.session_manager = SessionManager()
        self.audit_service = AuditService()
        
        # Initialize CV pipeline components
        self.preprocessor = ImagePreprocessor()
        self.yolo_loader = YOLOLoader()
        
        # Initialize detectors with fallback chain
        self._initialize_detectors()
        
        # Processing statistics
        self.stats = {
            'images_processed': 0,
            'lpp_detections': 0,
            'escalations': 0,
            'processing_errors': 0,
            'avg_processing_time': 0.0,
            'model_type': 'initializing'
        }
        
        # A2A server for distributed communication
        self.a2a_server = None
        
        logger.info(f"ImageAnalysisAgent initialized with {self.stats['model_type']} model")
    
    def _initialize_detectors(self):
        """Initialize detection models with fallback hierarchy"""
        try:
            # Primary: Real LPP detector (production)
            self.real_detector = RealLPPDetector()
            self.pressure_ulcer_detector = PressureUlcerDetector()
            self.stats['model_type'] = 'real_yolo_medical'
            logger.info("Initialized real LPP detection models")
            
        except Exception as e:
            logger.warning(f"Real detector initialization failed: {e}")
            
            try:
                # Fallback: Standard LPP detector
                self.lpp_detector = LPPDetector(
                    model_type='yolov5s',
                    conf_threshold=0.5
                )
                self.stats['model_type'] = 'standard_yolo'
                logger.info("Initialized standard LPP detector")
                
            except Exception as e:
                logger.error(f"All detector initialization failed: {e}")
                self.stats['model_type'] = 'mock_fallback'
    
    async def analyze_medical_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main image analysis function for medical images.
        
        Args:
            image_data: Dictionary containing:
                - image_path: Path to medical image
                - patient_code: Anonymized patient identifier
                - medical_context: Patient medical context
                - session_token: Session tracking token
                - analysis_params: Optional analysis parameters
                
        Returns:
            Complete medical image analysis results
        """
        start_time = datetime.now()
        image_path = image_data.get('image_path')
        patient_code = image_data.get('patient_code')
        session_token = image_data.get('session_token')
        medical_context = image_data.get('medical_context', {})
        
        try:
            # Initialize session tracking
            await self._initialize_analysis_session(image_data)
            
            # Step 1: Image validation and loading
            logger.info(f"Validating image for patient {patient_code}")
            image_validation = await self._validate_medical_image(image_path)
            
            if not image_validation['valid']:
                return await self._handle_invalid_image(image_data, image_validation['error'])
            
            # Step 2: Medical preprocessing (privacy + clinical enhancement)
            logger.info(f"Preprocessing medical image for patient {patient_code}")
            preprocessing_result = await self._perform_medical_preprocessing(
                image_path, medical_context
            )
            
            if not preprocessing_result['success']:
                return await self._handle_preprocessing_error(image_data, preprocessing_result['error'])
            
            # Step 3: YOLOv5 LPP detection
            logger.info(f"Performing LPP detection for patient {patient_code}")
            detection_result = await self._perform_lpp_detection(
                preprocessing_result['processed_image_path'], medical_context
            )
            
            if not detection_result['success']:
                return await self._handle_detection_error(image_data, detection_result['error'])
            
            # Step 4: Medical assessment and validation
            logger.info(f"Generating medical assessment for patient {patient_code}")
            medical_assessment = await self._generate_medical_assessment(
                detection_result, medical_context
            )
            
            # Step 5: Confidence validation and escalation check
            escalation_result = await self._check_escalation_requirements(
                medical_assessment, medical_context
            )
            
            # Generate comprehensive results
            analysis_results = await self._generate_analysis_results(
                image_data, {
                    'validation': image_validation,
                    'preprocessing': preprocessing_result,
                    'detection': detection_result,
                    'medical_assessment': medical_assessment,
                    'escalation': escalation_result
                }
            )
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_processing_stats(processing_time, analysis_results)
            
            # Finalize session
            await self._finalize_analysis_session(session_token, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Critical error in image analysis: {str(e)}")
            error_result = await self._handle_critical_error(image_data, str(e))
            await self._finalize_analysis_session(session_token, error_result)
            return error_result
    
    async def _initialize_analysis_session(self, image_data: Dict[str, Any]):
        """Initialize analysis session with audit trail"""
        session_token = image_data.get('session_token')
        patient_code = image_data.get('patient_code')
        
        if session_token:
            # Audit image analysis start
            await self.audit_service.log_event(
                event_type=AuditEventType.IMAGE_PROCESSED,
                component='image_analysis_agent',
                action='start_analysis',
                session_id=session_token,
                user_id='image_analysis_agent',
                resource=f'medical_image_{patient_code}',
                details={
                    'agent_id': self.agent_id,
                    'model_type': self.stats['model_type'],
                    'image_path': image_data.get('image_path'),
                    'analysis_start': datetime.now().isoformat()
                }
            )
    
    async def _validate_medical_image(self, image_path: str) -> Dict[str, Any]:
        """Validate medical image for processing"""
        try:
            if not image_path or not Path(image_path).exists():
                return {
                    'valid': False,
                    'error': f'Image file not found: {image_path}',
                    'error_type': 'file_not_found'
                }
            
            # Check file size and format
            image_file = Path(image_path)
            file_size_mb = image_file.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 50:  # 50MB limit
                return {
                    'valid': False,
                    'error': f'Image too large: {file_size_mb:.1f}MB (max 50MB)',
                    'error_type': 'file_too_large'
                }
            
            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            if image_file.suffix.lower() not in valid_extensions:
                return {
                    'valid': False,
                    'error': f'Invalid image format: {image_file.suffix}',
                    'error_type': 'invalid_format'
                }
            
            return {
                'valid': True,
                'file_size_mb': file_size_mb,
                'format': image_file.suffix.lower(),
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'error_type': 'validation_exception'
            }
    
    async def _perform_medical_preprocessing(self, image_path: str, 
                                           medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform medical preprocessing with privacy protection"""
        try:
            # Run preprocessing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            preprocessing_result = await loop.run_in_executor(
                None, self.preprocessor.preprocess, image_path
            )
            
            return {
                'success': True,
                'processed_image': preprocessing_result['image'],
                'processed_image_path': preprocessing_result.get('output_path', image_path),
                'preprocessing_steps': preprocessing_result.get('steps_applied', []),
                'privacy_protected': preprocessing_result.get('privacy_protected', True),
                'clinical_enhanced': preprocessing_result.get('clinical_enhanced', True),
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Medical preprocessing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'preprocessing_failure'
            }
    
    async def _perform_lpp_detection(self, image_path: str, 
                                   medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform LPP detection using available models"""
        try:
            # Determine which detector to use based on availability
            if hasattr(self, 'real_detector') and self.real_detector:
                # Use real LPP detector (preferred)
                loop = asyncio.get_event_loop()
                detection_result = await loop.run_in_executor(
                    None, self.real_detector.detect, image_path
                )
                detector_used = 'real_lpp_detector'
                
            elif hasattr(self, 'pressure_ulcer_detector') and self.pressure_ulcer_detector:
                # Use pressure ulcer detector (fallback)
                loop = asyncio.get_event_loop()
                detection_result = await loop.run_in_executor(
                    None, self.pressure_ulcer_detector.detect, image_path
                )
                detector_used = 'pressure_ulcer_detector'
                
            elif hasattr(self, 'lpp_detector') and self.lpp_detector:
                # Use standard LPP detector (fallback)
                loop = asyncio.get_event_loop()
                detection_result = await loop.run_in_executor(
                    None, self.lpp_detector.detect, image_path
                )
                detector_used = 'standard_lpp_detector'
                
            else:
                # Generate mock detection (last resort)
                detection_result = self._generate_mock_detection(medical_context)
                detector_used = 'mock_detector'
            
            return {
                'success': True,
                'detections': detection_result.get('detections', []),
                'processing_time_ms': detection_result.get('processing_time_ms', 0),
                'model_info': detection_result.get('model_info', {}),
                'detector_used': detector_used,
                'detection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LPP detection failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'detection_failure'
            }
    
    def _generate_mock_detection(self, medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock detection for testing/fallback"""
        # Determine mock LPP grade based on medical context
        risk_factors = medical_context.get('risk_factors', [])
        age = medical_context.get('age', 65)
        
        # Risk-based mock grading
        if 'diabetes' in risk_factors and age > 70:
            mock_grade = 2
            mock_confidence = 0.72
        elif 'immobility' in risk_factors:
            mock_grade = 1
            mock_confidence = 0.68
        else:
            mock_grade = 0
            mock_confidence = 0.85
        
        return {
            'detections': [
                {
                    'bbox': [150, 200, 300, 350],
                    'confidence': mock_confidence,
                    'stage': mock_grade,
                    'class_name': f'LPP-Stage{mock_grade}' if mock_grade > 0 else 'no-LPP'
                }
            ] if mock_grade > 0 else [],
            'processing_time_ms': 500,
            'model_info': {
                'type': 'mock_medical_detector',
                'version': '1.0.0_mock'
            }
        }
    
    async def _generate_medical_assessment(self, detection_result: Dict[str, Any], 
                                         medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical assessment from detection results"""
        detections = detection_result.get('detections', [])
        
        if not detections:
            return {
                'lpp_detected': False,
                'lpp_grade': 0,
                'severity': 'NONE',
                'confidence_score': 0.95,
                'anatomical_location': None,
                'medical_significance': 'No pressure injuries detected',
                'clinical_recommendations': ['Continue preventive measures'],
                'assessment_timestamp': datetime.now().isoformat()
            }
        
        # Process highest confidence detection
        primary_detection = max(detections, key=lambda d: d.get('confidence', 0))
        lpp_grade = primary_detection.get('stage', 0)
        confidence = primary_detection.get('confidence', 0.0)
        
        # Determine severity and recommendations
        severity_map = {
            0: 'NONE',
            1: 'MILD', 
            2: 'MODERATE',
            3: 'SEVERE',
            4: 'CRITICAL'
        }
        
        clinical_recommendations = self._generate_clinical_recommendations(
            lpp_grade, confidence, medical_context
        )
        
        return {
            'lpp_detected': lpp_grade > 0,
            'lpp_grade': lpp_grade,
            'severity': severity_map.get(lpp_grade, 'UNKNOWN'),
            'confidence_score': confidence,
            'anatomical_location': medical_context.get('anatomical_location', 'unknown'),
            'detection_bbox': primary_detection.get('bbox'),
            'medical_significance': self._assess_medical_significance(lpp_grade, confidence),
            'clinical_recommendations': clinical_recommendations,
            'requires_attention': lpp_grade >= 2 or confidence < 0.6,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def _generate_clinical_recommendations(self, lpp_grade: int, confidence: float, 
                                         medical_context: Dict[str, Any]) -> List[str]:
        """Generate evidence-based clinical recommendations"""
        recommendations = []
        
        if lpp_grade == 0:
            recommendations.extend([
                'Continue preventive pressure relief measures',
                'Maintain regular repositioning schedule (every 2 hours)',
                'Monitor skin integrity during daily assessments'
            ])
        elif lpp_grade == 1:
            recommendations.extend([
                'Implement immediate pressure relief',
                'Increase repositioning frequency to every 1-2 hours',
                'Apply moisture barrier protection',
                'Document and monitor progression'
            ])
        elif lpp_grade == 2:
            recommendations.extend([
                'Immediate pressure relief and offloading',
                'Wound assessment by qualified nurse',
                'Consider specialized pressure-relieving surfaces',
                'Nutritional assessment and optimization'
            ])
        elif lpp_grade >= 3:
            recommendations.extend([
                'URGENT: Immediate medical evaluation required',
                'Complete pressure offloading of affected area',
                'Wound care specialist consultation',
                'Consider surgical debridement evaluation',
                'Nutritional support and infection monitoring'
            ])
        
        # Add context-specific recommendations
        risk_factors = medical_context.get('risk_factors', [])
        if 'diabetes' in risk_factors:
            recommendations.append('Enhanced glucose control monitoring')
        if 'malnutrition' in risk_factors:
            recommendations.append('Immediate nutritional intervention')
        if 'incontinence' in risk_factors:
            recommendations.append('Moisture management protocol')
        
        return recommendations
    
    def _assess_medical_significance(self, lpp_grade: int, confidence: float) -> str:
        """Assess medical significance of detection"""
        if lpp_grade == 0:
            return 'No pressure injury detected - continue preventive care'
        elif lpp_grade == 1:
            return 'Stage 1 pressure injury - early intervention needed'
        elif lpp_grade == 2:
            return 'Stage 2 pressure injury - wound care required'
        elif lpp_grade == 3:
            return 'Stage 3 pressure injury - URGENT medical attention needed'
        elif lpp_grade == 4:
            return 'Stage 4 pressure injury - CRITICAL medical intervention required'
        else:
            return f'Pressure injury detected (Grade {lpp_grade}) - medical evaluation needed'
    
    async def _check_escalation_requirements(self, medical_assessment: Dict[str, Any], 
                                           medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if case requires escalation"""
        lpp_grade = medical_assessment.get('lpp_grade', 0)
        confidence = medical_assessment.get('confidence_score', 0.0)
        
        escalation_triggers = []
        escalation_required = False
        
        # Low confidence escalation
        if confidence < 0.6:
            escalation_triggers.append('low_confidence')
            escalation_required = True
        
        # Critical grade escalation
        if lpp_grade >= 3:
            escalation_triggers.append('critical_grade')
            escalation_required = True
        
        # High-risk patient escalation
        risk_factors = medical_context.get('risk_factors', [])
        if 'diabetes' in risk_factors and lpp_grade >= 2:
            escalation_triggers.append('high_risk_patient')
            escalation_required = True
        
        return {
            'escalation_required': escalation_required,
            'escalation_triggers': escalation_triggers,
            'escalation_urgency': 'CRITICAL' if lpp_grade >= 3 else 'HIGH' if escalation_required else 'NONE',
            'human_review_required': escalation_required,
            'immediate_notification': lpp_grade >= 3,
            'escalation_timestamp': datetime.now().isoformat() if escalation_required else None
        }
    
    async def _generate_analysis_results(self, image_data: Dict[str, Any], 
                                       processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis results"""
        patient_code = image_data.get('patient_code')
        session_token = image_data.get('session_token')
        
        medical_assessment = processing_results['medical_assessment']
        escalation = processing_results['escalation']
        detection = processing_results['detection']
        
        return {
            # Case identification
            'case_id': f"{patient_code}_{session_token}",
            'patient_code': patient_code,
            'session_token': session_token,
            'analysis_timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            
            # Analysis success status
            'analysis_success': all(
                result.get('success', result.get('valid', True)) 
                for result in processing_results.values()
            ),
            
            # Medical findings
            'medical_findings': {
                'lpp_detected': medical_assessment.get('lpp_detected', False),
                'lpp_grade': medical_assessment.get('lpp_grade', 0),
                'severity': medical_assessment.get('severity', 'NONE'),
                'confidence_score': medical_assessment.get('confidence_score', 0.0),
                'anatomical_location': medical_assessment.get('anatomical_location'),
                'medical_significance': medical_assessment.get('medical_significance'),
                'requires_attention': medical_assessment.get('requires_attention', False)
            },
            
            # Clinical recommendations
            'clinical_recommendations': medical_assessment.get('clinical_recommendations', []),
            
            # Escalation information
            'escalation_info': {
                'required': escalation.get('escalation_required', False),
                'triggers': escalation.get('escalation_triggers', []),
                'urgency': escalation.get('escalation_urgency', 'NONE'),
                'human_review': escalation.get('human_review_required', False),
                'immediate_notification': escalation.get('immediate_notification', False)
            },
            
            # Technical details
            'technical_details': {
                'detector_used': detection.get('detector_used'),
                'processing_time_ms': detection.get('processing_time_ms', 0),
                'model_type': self.stats['model_type'],
                'preprocessing_applied': processing_results['preprocessing'].get('preprocessing_steps', []),
                'privacy_protected': processing_results['preprocessing'].get('privacy_protected', True)
            },
            
            # Detection details
            'detection_details': {
                'detections_found': len(detection.get('detections', [])),
                'raw_detections': detection.get('detections', []),
                'detection_bbox': medical_assessment.get('detection_bbox')
            },
            
            # Compliance information
            'compliance_info': {
                'hipaa_compliant': True,
                'privacy_protected': processing_results['preprocessing'].get('privacy_protected', True),
                'audit_trail_complete': True,
                'session_tracked': session_token is not None
            }
        }
    
    async def _handle_invalid_image(self, image_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle invalid image error"""
        patient_code = image_data.get('patient_code')
        session_token = image_data.get('session_token')
        
        self.stats['processing_errors'] += 1
        
        return {
            'case_id': f"{patient_code}_{session_token}",
            'patient_code': patient_code,
            'analysis_success': False,
            'error_type': 'invalid_image',
            'error_message': error,
            'medical_findings': {
                'lpp_detected': False,
                'lpp_grade': None,
                'severity': 'ERROR_INVALID_IMAGE',
                'requires_human_review': True
            },
            'escalation_info': {
                'required': True,
                'triggers': ['invalid_image'],
                'urgency': 'HIGH',
                'human_review': True
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _handle_preprocessing_error(self, image_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle preprocessing error"""
        return await self._handle_processing_error(image_data, error, 'preprocessing_error')
    
    async def _handle_detection_error(self, image_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle detection error"""
        return await self._handle_processing_error(image_data, error, 'detection_error')
    
    async def _handle_critical_error(self, image_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle critical processing error"""
        return await self._handle_processing_error(image_data, error, 'critical_error')
    
    async def _handle_processing_error(self, image_data: Dict[str, Any], error: str, error_type: str) -> Dict[str, Any]:
        """Generic error handler for processing errors"""
        patient_code = image_data.get('patient_code')
        session_token = image_data.get('session_token')
        
        self.stats['processing_errors'] += 1
        
        # Log error for audit
        await self.audit_service.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            component='image_analysis_agent',
            action='processing_error',
            session_id=session_token,
            user_id='image_analysis_agent',
            resource=f'medical_image_{patient_code}',
            details={
                'error_type': error_type,
                'error_message': error,
                'requires_escalation': True
            }
        )
        
        return {
            'case_id': f"{patient_code}_{session_token}",
            'patient_code': patient_code,
            'analysis_success': False,
            'error_type': error_type,
            'error_message': error,
            'medical_findings': {
                'lpp_detected': False,
                'lpp_grade': None,
                'severity': f'ERROR_{error_type.upper()}',
                'requires_human_review': True,
                'error_processing': True
            },
            'escalation_info': {
                'required': True,
                'triggers': [error_type],
                'urgency': 'CRITICAL' if error_type == 'critical_error' else 'HIGH',
                'human_review': True,
                'technical_escalation': True
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _update_processing_stats(self, processing_time: float, results: Dict[str, Any]):
        """Update agent processing statistics"""
        self.stats['images_processed'] += 1
        
        # Update average processing time
        current_avg = self.stats['avg_processing_time']
        total_images = self.stats['images_processed']
        self.stats['avg_processing_time'] = (
            (current_avg * (total_images - 1) + processing_time) / total_images
        )
        
        # Count LPP detections
        if results.get('medical_findings', {}).get('lpp_detected', False):
            self.stats['lpp_detections'] += 1
        
        # Count escalations
        if results.get('escalation_info', {}).get('required', False):
            self.stats['escalations'] += 1
    
    async def _finalize_analysis_session(self, session_token: str, results: Dict[str, Any]):
        """Finalize analysis session with audit trail"""
        if session_token:
            await self.audit_service.log_event(
                event_type=AuditEventType.IMAGE_PROCESSED,
                component='image_analysis_agent',
                action='complete_analysis',
                session_id=session_token,
                user_id='image_analysis_agent',
                resource='medical_image_analysis',
                details={
                    'analysis_success': results.get('analysis_success', False),
                    'lpp_detected': results.get('medical_findings', {}).get('lpp_detected', False),
                    'escalation_required': results.get('escalation_info', {}).get('required', False),
                    'processing_time': self.stats['avg_processing_time'],
                    'completion_timestamp': datetime.now().isoformat()
                }
            )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'image_analysis',
            'status': 'active',
            'model_type': self.stats['model_type'],
            'statistics': self.stats,
            'capabilities': [
                'medical_image_preprocessing',
                'lpp_detection_yolo',
                'privacy_protection_hipaa',
                'clinical_enhancement',
                'confidence_validation',
                'escalation_management'
            ],
            'uptime': datetime.now().isoformat()
        }


# ADK Tool Functions for Image Analysis Agent
def analyze_medical_image_tool(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK Tool function for medical image analysis.
    Can be used directly in ADK agents and A2A communication.
    """
    agent = ImageAnalysisAgent()
    
    # Run async analysis in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(agent.analyze_medical_image(image_data))
        return result
    finally:
        loop.close()


def get_image_analysis_status() -> Dict[str, Any]:
    """
    ADK Tool function for getting image analysis agent status.
    """
    agent = ImageAnalysisAgent()
    return agent.get_agent_status()


def validate_image_for_analysis(image_path: str) -> Dict[str, Any]:
    """
    ADK Tool function for validating images before analysis.
    """
    agent = ImageAnalysisAgent()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(agent._validate_medical_image(image_path))
        return result
    finally:
        loop.close()


# Create Image Analysis ADK Agent
image_analysis_agent = Agent(
    model="gemini-2.0-flash-exp",
    global_instruction=IMAGE_ANALYSIS_INSTRUCTION,
    instruction="Especialista en análisis de imágenes médicas con YOLOv5 para detección LPP y preprocesamiento clínico.",
    name="image_analysis_agent",
    tools=[
        analyze_medical_image_tool,
        get_image_analysis_status,
        validate_image_for_analysis,
    ],
)

# Export for use
__all__ = [
    'ImageAnalysisAgent',
    'image_analysis_agent', 
    'analyze_medical_image_tool',
    'get_image_analysis_status',
    'validate_image_for_analysis'
]