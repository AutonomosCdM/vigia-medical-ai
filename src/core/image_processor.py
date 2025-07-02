"""
Procesador centralizado de imágenes para el sistema Vigía
"""
import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..cv_pipeline.detector import LPPDetector
from ..cv_pipeline.preprocessor import ImagePreprocessor
from ..utils.image_utils import (
    is_valid_image, 
    save_detection_visualization, 
    anonymize_image
)
from .constants import LPP_GRADE_DESCRIPTIONS, LPP_GRADE_RECOMMENDATIONS


class ImageProcessor:
    """
    Procesador unificado de imágenes que combina preprocesamiento y detección.
    Evita duplicación de código entre CLI y WhatsApp processor.
    """
    
    def __init__(self, 
                 model_type: str = 'yolov5s',
                 confidence_threshold: float = 0.25,
                 anonymize: bool = True):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            model_type: Tipo de modelo YOLO a usar
            confidence_threshold: Umbral de confianza para detecciones
            anonymize: Si se debe anonimizar rostros en las imágenes
        """
        self.logger = logging.getLogger('vigia.image_processor')
        self.anonymize = anonymize
        
        # Inicializar detector y preprocesador
        self.detector = LPPDetector(
            model_type=model_type,
            conf_threshold=confidence_threshold
        )
        self.preprocessor = ImagePreprocessor()
        
        self.logger.info(f"ImageProcessor initialized with model {model_type}")
    
    def process_image(self, 
                     image_path: str,
                     patient_code: Optional[str] = None,
                     save_visualization: bool = False,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesa una imagen completa: validación, preprocesamiento y detección.
        
        Args:
            image_path: Ruta a la imagen
            patient_code: Código del paciente (opcional)
            save_visualization: Si guardar visualización con detecciones
            output_dir: Directorio para guardar resultados
            
        Returns:
            Dict con resultados del procesamiento
        """
        try:
            # Validar imagen
            if not is_valid_image(image_path):
                return {
                    "success": False,
                    "error": "Invalid image file",
                    "image_path": image_path
                }
            
            # Preprocesar
            processed_img, metadata = self._preprocess_image(image_path)
            
            # Detectar
            detection_results = self.detector.detect(processed_img)
            
            # Enriquecer resultados
            enriched_results = self._enrich_results(
                detection_results, 
                metadata,
                patient_code
            )
            
            # Guardar visualización si se requiere
            if save_visualization and output_dir:
                viz_path = self._save_visualization(
                    image_path,
                    enriched_results,
                    output_dir
                )
                enriched_results["visualization_path"] = viz_path
            
            return {
                "success": True,
                "results": enriched_results,
                "image_path": image_path,
                "patient_code": patient_code
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def process_batch(self, 
                     image_paths: list,
                     patient_code: Optional[str] = None,
                     save_visualizations: bool = False,
                     output_dir: Optional[str] = None) -> list:
        """
        Procesa un lote de imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            patient_code: Código del paciente
            save_visualizations: Si guardar visualizaciones
            output_dir: Directorio de salida
            
        Returns:
            Lista de resultados de procesamiento
        """
        results = []
        
        for idx, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            
            result = self.process_image(
                image_path,
                patient_code,
                save_visualizations,
                output_dir
            )
            results.append(result)
        
        # Resumen
        successful = sum(1 for r in results if r.get("success", False))
        self.logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
        
        return results
    
    def _preprocess_image(self, image_path: str) -> Tuple[Any, Dict]:
        """Preprocesa la imagen y extrae metadata"""
        # Preprocesar
        processed_img = self.preprocessor.preprocess(image_path)
        
        # Extraer metadata
        metadata = {
            "original_path": image_path,
            "file_size": os.path.getsize(image_path),
            "file_name": os.path.basename(image_path)
        }
        
        # Anonimizar si es necesario
        if self.anonymize:
            processed_img = anonymize_image(processed_img)
        
        return processed_img, metadata
    
    def _enrich_results(self, 
                       detection_results: Dict,
                       metadata: Dict,
                       patient_code: Optional[str]) -> Dict:
        """
        Enriquece los resultados de detección con información adicional.
        """
        # Copiar resultados originales
        enriched = detection_results.copy()
        
        # Agregar metadata
        enriched["metadata"] = metadata
        enriched["patient_code"] = patient_code
        
        # Enriquecer cada detección
        if "detections" in enriched:
            for detection in enriched["detections"]:
                grade = detection.get("class", 0)
                
                # Agregar descripciones y recomendaciones
                detection["grade_description"] = LPP_GRADE_DESCRIPTIONS.get(
                    grade, 
                    "Grado no reconocido"
                )
                detection["recommendations"] = LPP_GRADE_RECOMMENDATIONS.get(
                    grade,
                    "Consultar con equipo médico"
                )
                
                # Calcular severidad
                detection["severity"] = self._calculate_severity(grade, detection.get("confidence", 0))
        
        # Agregar resumen
        enriched["summary"] = self._generate_summary(enriched)
        
        return enriched
    
    def _calculate_severity(self, grade: int, confidence: float) -> str:
        """Calcula la severidad basada en grado y confianza"""
        if grade == 0:
            return "low"
        elif grade == 1:
            return "medium"
        elif grade == 2:
            return "high" if confidence > 0.8 else "medium"
        else:  # grade 3 or 4
            return "critical"
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Genera un resumen de los resultados"""
        detections = results.get("detections", [])
        
        if not detections:
            return {
                "total_detections": 0,
                "max_grade": None,
                "requires_attention": False,
                "message": "No se detectaron lesiones por presión"
            }
        
        grades = [d.get("class", 0) for d in detections]
        max_grade = max(grades)
        
        return {
            "total_detections": len(detections),
            "max_grade": max_grade,
            "requires_attention": max_grade > 0,
            "grades_found": list(set(grades)),
            "message": f"Se detectaron {len(detections)} posibles lesiones. Grado máximo: {max_grade}"
        }
    
    def _save_visualization(self, 
                          image_path: str,
                          results: Dict,
                          output_dir: str) -> str:
        """Guarda visualización de las detecciones"""
        output_path = Path(output_dir) / f"viz_{Path(image_path).name}"
        
        save_detection_visualization(
            image_path,
            results.get("detections", []),
            str(output_path)
        )
        
        return str(output_path)