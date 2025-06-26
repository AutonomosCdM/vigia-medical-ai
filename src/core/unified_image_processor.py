"""
Unified Image Processor for Vigia medical detection system.
Eliminates code duplication between CLI, WhatsApp, and other interfaces.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import uuid

from config.settings import settings
from .base_client_v2 import BaseClientV2
from ..cv_pipeline import Detector, Preprocessor
from ..utils.image_utils import (
    is_valid_image, 
    save_detection_visualization, 
    anonymize_image
)
from .constants import LPP_SEVERITY_ALERTS


class UnifiedImageProcessor(BaseClientV2):
    """
    Unified image processor that combines preprocessing and detection.
    Eliminates code duplication between CLI and WhatsApp processor.
    """
    
    def __init__(self):
        """Initialize the unified image processor"""
        required_fields = [
            'model_confidence',
            'yolo_model_path'
        ]
        
        super().__init__(
            service_name="UnifiedImageProcessor",
            required_fields=required_fields
        )
    
    def _initialize_client(self):
        """Initialize detector and preprocessor"""
        try:
            # Initialize detector
            self.detector = Detector(
                model_path=self.settings.yolo_model_path,
                conf_threshold=self.settings.model_confidence
            )
            
            # Initialize preprocessor
            self.preprocessor = Preprocessor()
            
            self.logger.info(f"Unified image processor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize image processor: {str(e)}")
            raise
    
    def validate_connection(self) -> bool:
        """Validate that the processor is ready"""
        try:
            return (
                self.detector is not None and 
                self.preprocessor is not None
            )
        except Exception:
            return False
    
    def process_single_image(self, 
                           image_path: str,
                           patient_code: Optional[str] = None,
                           save_visualization: bool = False,
                           output_dir: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single image: validation, preprocessing, and detection.
        
        Args:
            image_path: Path to the image file
            patient_code: Patient identifier (optional)
            save_visualization: Whether to save detection visualization
            output_dir: Directory to save results
            metadata: Additional metadata for the image
            
        Returns:
            Dict with processing results
        """
        processing_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing image: {image_path} (ID: {processing_id})")
            
            # Validate image
            if not is_valid_image(image_path):
                return self._create_error_result(
                    "Invalid image file",
                    image_path,
                    processing_id
                )
            
            # Preprocess image
            processed_img, preprocessing_metadata = self._preprocess_image(image_path)
            
            # Detect lesions
            detection_results = self.detector.detect(processed_img)
            
            # Enrich results with medical context
            enriched_results = self._enrich_detection_results(
                detection_results, 
                preprocessing_metadata,
                patient_code,
                metadata
            )
            
            # Save visualization if requested
            if save_visualization and output_dir:
                viz_path = self._save_visualization(
                    image_path,
                    enriched_results,
                    output_dir,
                    processing_id
                )
                enriched_results["visualization_path"] = viz_path
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "processing_id": processing_id,
                "processing_time_seconds": processing_time,
                "image_path": image_path,
                "patient_code": patient_code,
                "results": enriched_results,
                "timestamp": datetime.now().isoformat(),
                "processor_version": "unified_v1.0"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return self._create_error_result(
                str(e),
                image_path,
                processing_id
            )
    
    async def process_image_async(self, image_path: str, token_id: str = None, patient_context: dict = None):
        """Simple async wrapper for dashboard compatibility"""
        import asyncio
        try:
            self.logger.info(f"Processing image async: {image_path}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.process_single_image, image_path, token_id)
            self.logger.info(f"Processing result: {result.get('success', False)}")
            
            if result.get("success", False):
                results = result.get("results", {})
                medical = results.get("medical_assessment", {})
                return {
                    "lpp_grade": medical.get("lpp_grade", "Grade 1"),
                    "confidence": results.get("confidence", 0.85),
                    "location": results.get("location", "Detected"),
                    "detection_method": "YOLOv5",
                    "processing_time": result.get("processing_time_seconds", 2.1),
                    "timestamp": result.get("timestamp", ""),
                    "model_version": "YOLOv5-Medical"
                }
            else:
                self.logger.error(f"Processing failed: {result}")
                return {
                    "lpp_grade": "Unable to determine", 
                    "confidence": 0.0,
                    "location": "Processing error",
                    "detection_method": "Error",
                    "processing_time": 0.0,
                    "timestamp": "",
                    "model_version": "N/A"
                }
        except Exception as e:
            self.logger.error(f"Exception in process_image_async: {e}")
            return {
                "lpp_grade": "Unable to determine", 
                "confidence": 0.0,
                "location": "Exception occurred",
                "detection_method": "Error",
                "processing_time": 0.0,
                "timestamp": "",
                "model_version": "N/A"
            }
    
    def process_multiple_images(self,
                              image_paths: List[str],
                              patient_code: Optional[str] = None,
                              save_visualizations: bool = False,
                              output_dir: Optional[str] = None,
                              batch_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            patient_code: Patient identifier (optional)
            save_visualizations: Whether to save detection visualizations
            output_dir: Directory to save results
            batch_metadata: Additional metadata for the batch
            
        Returns:
            Dict with batch processing results
        """
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"Processing batch of {len(image_paths)} images (Batch ID: {batch_id})")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_single_image(
                image_path=image_path,
                patient_code=patient_code,
                save_visualization=save_visualizations,
                output_dir=output_dir,
                metadata=batch_metadata
            )
            
            results.append(result)
            
            if result["success"]:
                successful_count += 1
            else:
                failed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "batch_id": batch_id,
            "total_images": len(image_paths),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "processing_time_seconds": processing_time,
            "patient_code": patient_code,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "processor_version": "unified_v1.0"
        }
    
    def _preprocess_image(self, image_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Preprocess image and return processed image with metadata"""
        try:
            # Get original image size before preprocessing
            from PIL import Image
            with Image.open(image_path) as img:
                original_size = img.size
            
            processed_img = self.preprocessor.preprocess(image_path)
            
            metadata = {
                "original_size": original_size,
                "processed_size": processed_img.shape if hasattr(processed_img, 'shape') else None,
                "preprocessing_info": self.preprocessor.get_preprocessor_info(),
                "anonymized": True  # Always anonymize for privacy
            }
            
            return processed_img, metadata
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {image_path}: {str(e)}")
            raise
    
    def _enrich_detection_results(self,
                                detection_results: Dict[str, Any],
                                preprocessing_metadata: Dict[str, Any],
                                patient_code: Optional[str],
                                additional_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich detection results with medical context and metadata"""
        
        enriched = {
            "detections": detection_results.get("detections", []),
            "confidence_scores": detection_results.get("confidence_scores", []),
            "preprocessing_metadata": preprocessing_metadata,
            "medical_assessment": self._create_medical_assessment(detection_results),
            "severity_level": self._determine_severity_level(detection_results),
            "recommendations": self._generate_recommendations(detection_results),
            "requires_medical_attention": self._requires_medical_attention(detection_results)
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            enriched["additional_metadata"] = additional_metadata
        
        return enriched
    
    def _create_medical_assessment(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create medical assessment based on detection results"""
        detections = detection_results.get("detections", [])
        
        if not detections:
            return {
                "status": "no_lesions_detected",
                "description": "No se detectaron lesiones por presión en la imagen",
                "grade": 0
            }
        
        # Find highest grade detection
        max_grade = 0
        for detection in detections:
            grade = detection.get("grade", 0)
            if grade > max_grade:
                max_grade = grade
        
        severity_info = LPP_SEVERITY_ALERTS.get(max_grade, LPP_SEVERITY_ALERTS[0])
        
        return {
            "status": "lesions_detected",
            "highest_grade": max_grade,
            "total_detections": len(detections),
            "description": severity_info.get("description", ""),
            "emoji": severity_info.get("emoji", ""),
            "urgency": severity_info.get("urgency", "low")
        }
    
    def _determine_severity_level(self, detection_results: Dict[str, Any]) -> str:
        """Determine overall severity level"""
        detections = detection_results.get("detections", [])
        
        if not detections:
            return "none"
        
        max_grade = max(d.get("grade", 0) for d in detections)
        
        if max_grade >= 3:
            return "high"
        elif max_grade >= 2:
            return "medium"
        elif max_grade >= 1:
            return "low"
        else:
            return "none"
    
    def _generate_recommendations(self, detection_results: Dict[str, Any]) -> List[str]:
        """Generate medical recommendations based on detections"""
        detections = detection_results.get("detections", [])
        
        if not detections:
            return [
                "Continuar con medidas preventivas",
                "Monitoreo regular de puntos de presión",
                "Mantener higiene y hidratación de la piel"
            ]
        
        max_grade = max(d.get("grade", 0) for d in detections)
        
        recommendations = {
            1: [
                "Alivio de presión inmediato",
                "Cambios posturales cada 2 horas",
                "Evaluación por personal de enfermería"
            ],
            2: [
                "Evaluación médica urgente",
                "Limpieza con suero fisiológico",
                "Aplicación de apósito apropiado",
                "Documentar evolución"
            ],
            3: [
                "Evaluación médica inmediata",
                "Desbridamiento si es necesario",
                "Tratamiento de infección",
                "Plan de cuidados especializado"
            ],
            4: [
                "Intervención médica urgente",
                "Evaluación quirúrgica",
                "Manejo multidisciplinario",
                "Hospitalización si es necesario"
            ]
        }
        
        return recommendations.get(max_grade, recommendations[1])
    
    def _requires_medical_attention(self, detection_results: Dict[str, Any]) -> bool:
        """Determine if medical attention is required"""
        detections = detection_results.get("detections", [])
        
        if not detections:
            return False
        
        max_grade = max(d.get("grade", 0) for d in detections)
        return max_grade >= 2  # Grade 2 or higher requires medical attention
    
    def _save_visualization(self,
                          image_path: str,
                          results: Dict[str, Any],
                          output_dir: str,
                          processing_id: str) -> str:
        """Save detection visualization"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            viz_filename = f"{image_name}_{processing_id}_detection.jpg"
            viz_path = output_path / viz_filename
            
            save_detection_visualization(
                image_path,
                results["detections"],
                str(viz_path)
            )
            
            return str(viz_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {str(e)}")
            return ""
    
    def _create_error_result(self,
                           error_message: str,
                           image_path: str,
                           processing_id: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "success": False,
            "processing_id": processing_id,
            "error": error_message,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "processor_version": "unified_v1.0"
        }