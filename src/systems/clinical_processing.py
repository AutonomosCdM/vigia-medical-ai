"""
Clinical Processing System - Capa 3: Sistema Especializado
Procesamiento de imágenes clínicas con detección de LPP.

Responsabilidades:
- Procesar imágenes médicas de forma segura
- Detectar y clasificar lesiones por presión (LPP)
- Validar contexto clínico
- Generar reportes médicos estructurados
- Mantener trazabilidad completa
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import hashlib

from ..core.input_packager import StandardizedInput
from ..core.medical_dispatcher import TriageDecision
from ..core.phi_tokenization_client import TokenizedPatient
from ..cv_pipeline.detector import LPPDetector
from ..cv_pipeline.preprocessor import ImagePreprocessor
from ..db.supabase_client import SupabaseClient
from ..storage.medical_image_storage import MedicalImageStorage, AnatomicalRegion, ImageType
from ..utils.secure_logger import SecureLogger
from ..utils.validators import validate_patient_code_format

logger = SecureLogger("clinical_processing_system")


class LPPGrade(Enum):
    """Clasificación de grados de LPP según NPUAP/EPUAP."""
    GRADE_1 = 1  # Eritema no blanqueable
    GRADE_2 = 2  # Pérdida parcial de dermis
    GRADE_3 = 3  # Pérdida total del grosor de la piel
    GRADE_4 = 4  # Pérdida total del tejido
    UNSTAGEABLE = 5  # No clasificable
    DEEP_TISSUE = 6  # Sospecha de lesión profunda


class ProcessingStatus(Enum):
    """Estados de procesamiento clínico."""
    RECEIVED = "received"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    DETECTING = "detecting"
    ANALYZING = "analyzing"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ClinicalDetection:
    """Resultado de detección clínica."""
    detected: bool
    lpp_grade: Optional[LPPGrade]
    confidence: float
    bounding_boxes: List[List[int]]
    clinical_features: Dict[str, Any]
    risk_factors: List[str]
    recommended_interventions: List[str]
    measurement_data: Optional[Dict[str, float]]  # área, perímetro, etc.


@dataclass
class ClinicalReport:
    """Reporte clínico estructurado con datos tokenizados (NO PHI)."""
    session_id: str
    tokenized_patient: TokenizedPatient  # NO PHI - solo datos tokenizados
    detection_result: ClinicalDetection
    processing_time: float
    image_metadata: Dict[str, Any]
    clinical_notes: str
    quality_metrics: Dict[str, float]
    compliance_flags: List[str]
    timestamp: datetime
    processor_version: str


class ClinicalProcessingSystem:
    """
    Sistema especializado para procesamiento de imágenes clínicas.
    """
    
    def __init__(self, 
                 detector: Optional[LPPDetector] = None,
                 preprocessor: Optional[ImagePreprocessor] = None,
                 db_client: Optional[SupabaseClient] = None,
                 image_storage: Optional[MedicalImageStorage] = None):
        """
        Inicializar sistema de procesamiento clínico.
        
        Args:
            detector: Detector de LPP
            preprocessor: Preprocesador de imágenes
            db_client: Cliente de base de datos
            image_storage: Servicio de almacenamiento de imágenes médicas
        """
        self.detector = detector or LPPDetector()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.db_client = db_client
        self.image_storage = image_storage or MedicalImageStorage()
        
        # Configuración clínica
        self.min_confidence_threshold = 0.7
        self.quality_requirements = {
            "min_resolution": (224, 224),
            "max_resolution": (4096, 4096),
            "allowed_formats": [".jpg", ".jpeg", ".png", ".webp"],
            "max_file_size": 10 * 1024 * 1024  # 10MB
        }
        
        # Cache temporal para imágenes procesadas
        self.temp_dir = Path(tempfile.gettempdir()) / "vigia_clinical"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Versión del procesador
        self.processor_version = "1.0.0"
        
        logger.audit("clinical_processing_system_initialized", {
            "component": "layer3_clinical_processing",
            "processor_version": self.processor_version,
            "quality_requirements": self.quality_requirements,
            "medical_compliance": True
        })
    
    async def process(self, 
                     standardized_input: StandardizedInput,
                     triage_decision: TriageDecision) -> Dict[str, Any]:
        """
        Procesar imagen clínica según decisión de triage.
        
        Args:
            standardized_input: Input estandarizado
            triage_decision: Decisión del triage engine
            
        Returns:
            Dict con resultado del procesamiento clínico
        """
        session_id = standardized_input.session_id
        start_time = datetime.now(timezone.utc)
        status = ProcessingStatus.RECEIVED
        
        try:
            # Actualizar estado: VALIDATING
            status = ProcessingStatus.VALIDATING
            validation_result = await self._validate_clinical_input(standardized_input, triage_decision)
            if not validation_result["valid"]:
                raise ValueError(f"Clinical validation failed: {validation_result['reason']}")
            
            # Obtener datos tokenizados (NO PHI) desde triage decision
            tokenized_patient = validation_result["tokenized_patient"]
            
            # Obtener imagen
            image_path = await self._retrieve_clinical_image(standardized_input)
            
            # Actualizar estado: PREPROCESSING
            status = ProcessingStatus.PREPROCESSING
            preprocessed_data = await self._preprocess_clinical_image(image_path)
            
            # Store medical image in patient database
            await self._store_medical_image(
                preprocessed_data["image_path"],
                tokenized_patient,
                standardized_input
            )
            
            # Actualizar estado: DETECTING
            status = ProcessingStatus.DETECTING
            detection_result = await self._detect_pressure_injury(
                preprocessed_data["image_path"],
                preprocessed_data["metadata"]
            )
            
            # Actualizar estado: ANALYZING
            status = ProcessingStatus.ANALYZING
            clinical_analysis = await self._analyze_clinical_findings(
                detection_result,
                preprocessed_data["metadata"]
            )
            
            # Actualizar estado: GENERATING_REPORT
            status = ProcessingStatus.GENERATING_REPORT
            clinical_report = await self._generate_clinical_report(
                session_id,
                tokenized_patient,
                clinical_analysis,
                preprocessed_data["metadata"],
                start_time
            )
            
            # Guardar en base de datos si está disponible
            if self.db_client:
                await self._save_clinical_results(clinical_report)
            
            # Actualizar estado: COMPLETED
            status = ProcessingStatus.COMPLETED
            
            # Limpiar archivos temporales
            await self._cleanup_temp_files(session_id)
            
            # Log exitoso (sin PII)
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.audit("clinical_processing_completed", {
                "session_id": session_id,
                "status": status.value,
                "processing_time": processing_time,
                "lpp_detected": clinical_report.detection_result.detected,
                "lpp_grade": clinical_report.detection_result.lpp_grade.value if clinical_report.detection_result.lpp_grade else None,
                "confidence": clinical_report.detection_result.confidence
            })
            
            return {
                "success": True,
                "clinical_report": self._serialize_clinical_report(clinical_report),
                "processing_time": processing_time,
                "next_steps": self._determine_next_steps(clinical_report),
                "notifications_required": self._determine_notifications(clinical_report)
            }
            
        except Exception as e:
            logger.error("clinical_processing_failed", {
                "session_id": session_id,
                "status": status.value,
                "error": str(e),
                "processing_stage": status.value
            })
            
            # Limpiar archivos temporales en caso de error
            await self._cleanup_temp_files(session_id)
            
            return {
                "success": False,
                "error": str(e),
                "processing_stage": status.value,
                "recovery_suggestions": self._get_recovery_suggestions(status, str(e))
            }
    
    async def _validate_clinical_input(self, standardized_input: StandardizedInput, 
                                      triage_decision: TriageDecision) -> Dict[str, Any]:
        """Validar input para procesamiento clínico con datos tokenizados."""
        try:
            # Verificar que hay imagen
            if not standardized_input.metadata.get("has_media"):
                return {
                    "valid": False,
                    "reason": "No image provided for clinical processing"
                }
            
            # Verificar que tenemos datos tokenizados del triage
            if not hasattr(triage_decision, 'tokenized_patient') or not triage_decision.tokenized_patient:
                return {
                    "valid": False,
                    "reason": "No tokenized patient data from triage decision"
                }
            
            tokenized_patient = triage_decision.tokenized_patient
            
            # Verificar que los datos tokenizados son válidos
            if not tokenized_patient.token_id or not tokenized_patient.patient_alias:
                return {
                    "valid": False,
                    "reason": "Invalid tokenized patient data - missing token_id or alias"
                }
            
            # Verificar formato de imagen
            media_type = standardized_input.raw_content.get("media_type", "")
            if not self._is_valid_image_format(media_type):
                return {
                    "valid": False,
                    "reason": f"Unsupported image format: {media_type}"
                }
            
            # Verificar tamaño
            media_size = standardized_input.raw_content.get("media_size", 0)
            if media_size > self.quality_requirements["max_file_size"]:
                return {
                    "valid": False,
                    "reason": f"Image too large: {media_size} bytes"
                }
            
            logger.audit("clinical_input_validated", {
                "session_id": standardized_input.session_id,
                "patient_alias": tokenized_patient.patient_alias,  # NO PHI
                "token_id": tokenized_patient.token_id,
                "media_type": media_type,
                "phi_protected": True
            })
            
            return {
                "valid": True,
                "tokenized_patient": tokenized_patient,  # NO PHI
                "media_type": media_type,
                "media_size": media_size
            }
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    async def _retrieve_clinical_image(self, standardized_input: StandardizedInput) -> str:
        """Recuperar imagen clínica de forma segura."""
        session_id = standardized_input.session_id
        media_url = standardized_input.raw_content.get("media_url")
        
        if not media_url:
            raise ValueError("No media URL found in input")
        
        # Crear path temporal único
        file_extension = self._get_file_extension(standardized_input.raw_content.get("media_type", ""))
        temp_filename = f"{session_id}_{hashlib.sha256(media_url.encode()).hexdigest()[:8]}{file_extension}"
        temp_path = self.temp_dir / temp_filename
        
        # Descargar imagen
        try:
            await download_image_from_url(media_url, str(temp_path))
            
            # Verificar que el archivo existe y es válido
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise ValueError("Downloaded image is empty or invalid")
            
            logger.audit("clinical_image_retrieved", {
                "session_id": session_id,
                "image_size": temp_path.stat().st_size,
                "temp_path": str(temp_path)
            })
            
            return str(temp_path)
            
        except Exception as e:
            logger.error("image_retrieval_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            raise
    
    async def _preprocess_clinical_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocesar imagen clínica."""
        try:
            # Anonimizar imagen (remover metadata EXIF, etc.)
            anonymized_path = anonymize_image(image_path)
            
            # Preprocesar para detección
            preprocessed = self.preprocessor.preprocess(anonymized_path)
            
            # Extraer metadata de calidad
            quality_metrics = self.preprocessor.assess_quality(anonymized_path)
            
            return {
                "image_path": preprocessed["output_path"],
                "original_path": image_path,
                "metadata": {
                    "original_resolution": preprocessed.get("original_resolution"),
                    "processed_resolution": preprocessed.get("processed_resolution"),
                    "quality_metrics": quality_metrics,
                    "preprocessing_applied": preprocessed.get("operations", [])
                }
            }
            
        except Exception as e:
            logger.error("image_preprocessing_failed", {
                "image_path": image_path,
                "error": str(e)
            })
            raise
    
    async def _detect_pressure_injury(self, image_path: str, metadata: Dict[str, Any]) -> ClinicalDetection:
        """Detectar lesiones por presión en imagen."""
        try:
            # Ejecutar detección
            detection_results = self.detector.detect(image_path)
            
            # Procesar resultados
            if not detection_results or len(detection_results) == 0:
                return ClinicalDetection(
                    detected=False,
                    lpp_grade=None,
                    confidence=0.0,
                    bounding_boxes=[],
                    clinical_features={},
                    risk_factors=[],
                    recommended_interventions=[],
                    measurement_data=None
                )
            
            # Tomar detección con mayor confianza
            best_detection = max(detection_results, key=lambda d: d.get("confidence", 0))
            
            # Mapear grado de LPP
            lpp_grade = self._map_detection_to_lpp_grade(best_detection)
            
            # Extraer características clínicas
            clinical_features = self._extract_clinical_features(best_detection)
            
            # Identificar factores de riesgo
            risk_factors = self._identify_risk_factors(clinical_features, metadata)
            
            # Recomendar intervenciones
            interventions = self._recommend_interventions(lpp_grade, risk_factors)
            
            # Calcular medidas si es posible
            measurement_data = self._calculate_measurements(
                best_detection.get("bbox", []),
                metadata.get("processed_resolution")
            )
            
            return ClinicalDetection(
                detected=True,
                lpp_grade=lpp_grade,
                confidence=best_detection.get("confidence", 0.0),
                bounding_boxes=[best_detection.get("bbox", [])],
                clinical_features=clinical_features,
                risk_factors=risk_factors,
                recommended_interventions=interventions,
                measurement_data=measurement_data
            )
            
        except Exception as e:
            logger.error("lpp_detection_failed", {
                "image_path": image_path,
                "error": str(e)
            })
            raise
    
    async def _analyze_clinical_findings(self, 
                                       detection: ClinicalDetection,
                                       image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar hallazgos clínicos en profundidad."""
        try:
            analysis = {
                "detection": detection,
                "clinical_severity": self._assess_clinical_severity(detection),
                "urgency_level": self._determine_clinical_urgency(detection),
                "quality_assessment": self._assess_detection_quality(
                    detection, 
                    image_metadata.get("quality_metrics", {})
                ),
                "documentation_completeness": self._assess_documentation(detection, image_metadata),
                "follow_up_required": self._determine_follow_up_needs(detection)
            }
            
            return analysis
            
        except Exception as e:
            logger.error("clinical_analysis_failed", {"error": str(e)})
            raise
    
    async def _generate_clinical_report(self,
                                      session_id: str,
                                      tokenized_patient: TokenizedPatient,
                                      clinical_analysis: Dict[str, Any],
                                      image_metadata: Dict[str, Any],
                                      start_time: datetime) -> ClinicalReport:
        """Generar reporte clínico estructurado."""
        try:
            detection = clinical_analysis["detection"]
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Generar notas clínicas
            clinical_notes = self._generate_clinical_notes(clinical_analysis)
            
            # Determinar flags de compliance
            compliance_flags = self._check_compliance_requirements(
                detection,
                image_metadata,
                clinical_notes
            )
            
            report = ClinicalReport(
                session_id=session_id,
                tokenized_patient=tokenized_patient,  # NO PHI - solo datos tokenizados
                detection_result=detection,
                processing_time=processing_time,
                image_metadata=image_metadata,
                clinical_notes=clinical_notes,
                quality_metrics={
                    "image_quality": image_metadata.get("quality_metrics", {}).get("overall_quality", 0),
                    "detection_confidence": detection.confidence,
                    "documentation_score": clinical_analysis.get("documentation_completeness", 0)
                },
                compliance_flags=compliance_flags,
                timestamp=datetime.now(timezone.utc),
                processor_version=self.processor_version
            )
            
            return report
            
        except Exception as e:
            logger.error("report_generation_failed", {
                "session_id": session_id,
                "error": str(e)
            })
            raise
    
    async def _save_clinical_results(self, report: ClinicalReport):
        """Guardar resultados clínicos en base de datos."""
        if not self.db_client:
            return
        
        try:
            # Preparar datos para almacenamiento (solo datos tokenizados - NO PHI)
            detection_data = {
                "session_id": report.session_id,
                "token_id": report.tokenized_patient.token_id,  # NO PHI
                "patient_alias": report.tokenized_patient.patient_alias,  # NO PHI (Batman)
                "detected": report.detection_result.detected,
                "lpp_grade": report.detection_result.lpp_grade.value if report.detection_result.lpp_grade else None,
                "confidence": report.detection_result.confidence,
                "clinical_features": report.detection_result.clinical_features,
                "risk_factors": report.detection_result.risk_factors,
                "recommended_interventions": report.detection_result.recommended_interventions,
                "processing_time": report.processing_time,
                "quality_metrics": report.quality_metrics,
                "compliance_flags": report.compliance_flags,
                "processor_version": report.processor_version,
                "created_at": report.timestamp.isoformat(),
                "phi_protected": True  # Flag para confirmar que no hay PHI
            }
            
            # Guardar detección
            await self.db_client.create_detection(detection_data)
            
            # Guardar notas clínicas por separado (más seguro, solo datos tokenizados)
            notes_data = {
                "session_id": report.session_id,
                "token_id": report.tokenized_patient.token_id,  # NO PHI
                "patient_alias": report.tokenized_patient.patient_alias,  # NO PHI
                "clinical_notes": report.clinical_notes,
                "created_at": report.timestamp.isoformat(),
                "created_by": "clinical_processing_system",
                "phi_protected": True
            }
            
            await self.db_client.create_clinical_note(notes_data)
            
            logger.audit("clinical_results_saved", {
                "session_id": report.session_id,
                "detection_saved": True,
                "notes_saved": True
            })
            
        except Exception as e:
            logger.error("save_clinical_results_failed", {
                "session_id": report.session_id,
                "error": str(e)
            })
            # No re-throw - guardar en DB es opcional
    
    async def _store_medical_image(
        self,
        image_path: str,
        tokenized_patient: TokenizedPatient,
        standardized_input: StandardizedInput
    ):
        """
        Store medical image in patient database with metadata
        
        Args:
            image_path: Path to processed image
            tokenized_patient: Tokenized patient data (NO PHI)
            standardized_input: Original input with metadata
        """
        try:
            # Extract anatomical region from input metadata
            anatomical_region = self._extract_anatomical_region(standardized_input)
            
            # Determine image type based on context
            image_type = self._determine_image_type(standardized_input)
            
            # Generate clinical context (PHI-safe)
            clinical_context = self._generate_clinical_context(standardized_input, tokenized_patient)
            
            # Store image in patient database
            image_record = await self.image_storage.store_medical_image(
                image_file_path=image_path,
                tokenized_patient=tokenized_patient,
                anatomical_region=anatomical_region,
                image_type=image_type,
                clinical_context=clinical_context,
                uploaded_by="clinical_processing_system"
            )
            
            logger.audit("medical_image_stored_during_processing", {
                "session_id": standardized_input.session_id,
                "image_id": image_record.image_id,
                "patient_alias": tokenized_patient.patient_alias,
                "anatomical_region": anatomical_region.value,
                "image_type": image_type.value
            })
            
        except Exception as e:
            logger.error("medical_image_storage_failed", {
                "session_id": standardized_input.session_id,
                "patient_alias": tokenized_patient.patient_alias,
                "error": str(e)
            })
            # Don't fail the entire processing pipeline if image storage fails
    
    def _extract_anatomical_region(self, standardized_input: StandardizedInput) -> AnatomicalRegion:
        """Extract anatomical region from input metadata"""
        
        # Try to extract from text content first
        text_content = standardized_input.text_content.lower() if standardized_input.text_content else ""
        
        # Simple keyword matching for anatomical regions
        region_keywords = {
            AnatomicalRegion.SACRUM: ["sacro", "sacrum", "coxis", "coccyx"],
            AnatomicalRegion.HEEL: ["talon", "heel", "calcáneo", "calcaneus"],
            AnatomicalRegion.ELBOW: ["codo", "elbow", "olécranon", "olecranon"],
            AnatomicalRegion.SHOULDER_BLADE: ["omóplato", "shoulder", "escápula", "scapula"],
            AnatomicalRegion.HIP: ["cadera", "hip", "trocánter", "trochanter"],
            AnatomicalRegion.ANKLE: ["tobillo", "ankle", "maléolo", "malleolus"],
            AnatomicalRegion.KNEE: ["rodilla", "knee", "patela", "patella"]
        }
        
        for region, keywords in region_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                return region
        
        # Default to sacrum (most common pressure injury location)
        return AnatomicalRegion.SACRUM
    
    def _determine_image_type(self, standardized_input: StandardizedInput) -> ImageType:
        """Determine image type based on context"""
        
        text_content = standardized_input.text_content.lower() if standardized_input.text_content else ""
        
        # Check for follow-up indicators
        followup_keywords = ["seguimiento", "control", "evolución", "progress", "follow-up", "followup"]
        if any(keyword in text_content for keyword in followup_keywords):
            return ImageType.WOUND_PROGRESS
        
        # Check for post-treatment indicators
        treatment_keywords = ["tratamiento", "después", "post", "treatment", "after", "curación"]
        if any(keyword in text_content for keyword in treatment_keywords):
            return ImageType.POST_TREATMENT
        
        # Check for baseline indicators
        baseline_keywords = ["inicial", "baseline", "primera", "first", "nuevo", "new"]
        if any(keyword in text_content for keyword in baseline_keywords):
            return ImageType.BASELINE_SKIN
        
        # Default to pressure injury assessment
        return ImageType.PRESSURE_INJURY_ASSESSMENT
    
    def _generate_clinical_context(
        self,
        standardized_input: StandardizedInput,
        tokenized_patient: TokenizedPatient
    ) -> str:
        """Generate PHI-safe clinical context"""
        
        context_parts = []
        
        # Add image type context
        image_type = self._determine_image_type(standardized_input)
        context_parts.append(f"Image type: {image_type.value.replace('_', ' ')}")
        
        # Add anatomical region
        region = self._extract_anatomical_region(standardized_input)
        context_parts.append(f"Location: {region.value}")
        
        # Add patient age range (tokenized, not exact age)
        context_parts.append(f"Patient age range: {tokenized_patient.age_range}")
        
        # Add risk factors (from tokenized data)
        if tokenized_patient.risk_factors:
            active_risks = [k for k, v in tokenized_patient.risk_factors.items() if v]
            if active_risks:
                context_parts.append(f"Risk factors: {', '.join(active_risks)}")
        
        # Add sanitized text content (remove any potential PHI)
        if standardized_input.text_content:
            sanitized_text = self._sanitize_clinical_text(standardized_input.text_content)
            if sanitized_text:
                context_parts.append(f"Clinical notes: {sanitized_text}")
        
        return " | ".join(context_parts)
    
    def _sanitize_clinical_text(self, text: str) -> str:
        """Remove potential PHI from clinical text"""
        import re
        
        # Remove common PHI patterns
        sanitized = text
        
        # Remove specific names (simple approach)
        sanitized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT]', sanitized)
        
        # Remove phone numbers
        sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', sanitized)
        
        # Remove dates
        sanitized = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', sanitized)
        
        # Remove medical record numbers
        sanitized = re.sub(r'\b(MRN|ID|#)\s*:?\s*\w+\b', '[MRN]', sanitized, flags=re.IGNORECASE)
        
        # Limit length and keep only relevant medical content
        sanitized = sanitized[:200] + "..." if len(sanitized) > 200 else sanitized
        
        return sanitized.strip()
    
    async def _cleanup_temp_files(self, session_id: str):
        """Limpiar archivos temporales de forma segura."""
        try:
            # Buscar todos los archivos de la sesión
            for temp_file in self.temp_dir.glob(f"{session_id}*"):
                try:
                    temp_file.unlink()
                    logger.debug("temp_file_cleaned", {
                        "session_id": session_id,
                        "file": temp_file.name
                    })
                except Exception as e:
                    logger.warning("temp_file_cleanup_failed", {
                        "file": str(temp_file),
                        "error": str(e)
                    })
                    
        except Exception as e:
            logger.error("cleanup_failed", {
                "session_id": session_id,
                "error": str(e)
            })
    
    def _extract_patient_code(self, text: str) -> Optional[str]:
        """Extraer código de paciente del texto."""
        if not text:
            return None
        
        import re
        pattern = r'\b[A-Z]{2}-\d{4}-\d{3}\b'
        matches = re.findall(pattern, text.upper())
        
        return matches[0] if matches else None
    
    def _is_valid_image_format(self, media_type: str) -> bool:
        """Verificar si el formato de imagen es válido."""
        valid_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        return media_type.lower() in valid_types
    
    def _get_file_extension(self, media_type: str) -> str:
        """Obtener extensión de archivo desde media type."""
        mapping = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp"
        }
        return mapping.get(media_type.lower(), ".jpg")
    
    def _map_detection_to_lpp_grade(self, detection: Dict[str, Any]) -> LPPGrade:
        """Mapear resultado de detección a grado de LPP."""
        class_id = detection.get("class_id", 0)
        class_name = detection.get("class_name", "").lower()
        
        # Mapeo basado en clase detectada
        grade_mapping = {
            "grade1": LPPGrade.GRADE_1,
            "grade2": LPPGrade.GRADE_2,
            "grade3": LPPGrade.GRADE_3,
            "grade4": LPPGrade.GRADE_4,
            "unstageable": LPPGrade.UNSTAGEABLE,
            "deep_tissue": LPPGrade.DEEP_TISSUE
        }
        
        # Intentar mapear por nombre de clase
        for key, grade in grade_mapping.items():
            if key in class_name:
                return grade
        
        # Mapeo por ID si está disponible
        if 1 <= class_id <= 6:
            return LPPGrade(class_id)
        
        # Default a grado 2 si no se puede determinar
        return LPPGrade.GRADE_2
    
    def _extract_clinical_features(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer características clínicas de la detección."""
        return {
            "location": detection.get("location", "unknown"),
            "size_category": detection.get("size_category", "medium"),
            "tissue_involvement": detection.get("tissue_involvement", "dermis"),
            "color_characteristics": detection.get("color", "erythema"),
            "border_definition": detection.get("border", "defined"),
            "exudate_present": detection.get("exudate", False),
            "necrosis_present": detection.get("necrosis", False),
            "infection_signs": detection.get("infection_signs", False)
        }
    
    def _identify_risk_factors(self, features: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Identificar factores de riesgo clínicos."""
        risk_factors = []
        
        # Basado en características
        if features.get("necrosis_present"):
            risk_factors.append("tissue_necrosis")
        
        if features.get("infection_signs"):
            risk_factors.append("infection_risk")
        
        if features.get("size_category") in ["large", "extensive"]:
            risk_factors.append("large_wound_area")
        
        # Basado en localización
        high_risk_locations = ["sacrum", "heel", "ischium", "trochanter"]
        if any(loc in features.get("location", "").lower() for loc in high_risk_locations):
            risk_factors.append("high_risk_anatomical_location")
        
        return risk_factors
    
    def _recommend_interventions(self, lpp_grade: Optional[LPPGrade], risk_factors: List[str]) -> List[str]:
        """Recomendar intervenciones basadas en grado y factores de riesgo."""
        interventions = []
        
        if not lpp_grade:
            return ["clinical_reassessment_required"]
        
        # Intervenciones por grado
        grade_interventions = {
            LPPGrade.GRADE_1: [
                "pressure_relief_immediately",
                "skin_moisturization",
                "position_changes_q2h",
                "nutritional_assessment"
            ],
            LPPGrade.GRADE_2: [
                "wound_dressing_hydrocolloid",
                "pressure_redistribution_surface",
                "pain_management",
                "infection_monitoring"
            ],
            LPPGrade.GRADE_3: [
                "specialized_wound_care",
                "surgical_consultation",
                "advanced_dressings",
                "nutritional_support"
            ],
            LPPGrade.GRADE_4: [
                "immediate_surgical_referral",
                "advanced_wound_therapy",
                "infection_control_protocol",
                "multidisciplinary_team_consult"
            ]
        }
        
        interventions.extend(grade_interventions.get(lpp_grade, []))
        
        # Intervenciones adicionales por factores de riesgo
        if "infection_risk" in risk_factors:
            interventions.append("antibiotic_consideration")
        
        if "tissue_necrosis" in risk_factors:
            interventions.append("debridement_assessment")
        
        return list(set(interventions))  # Eliminar duplicados
    
    def _calculate_measurements(self, bbox: List[int], resolution: Optional[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Calcular medidas aproximadas de la lesión."""
        if not bbox or len(bbox) < 4 or not resolution:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            
            # Estimación aproximada (asumiendo calibración estándar)
            # En producción, esto requeriría calibración real
            pixel_to_cm = 0.05  # Aproximación
            
            width_cm = width_pixels * pixel_to_cm
            height_cm = height_pixels * pixel_to_cm
            area_cm2 = width_cm * height_cm
            
            return {
                "width_cm": round(width_cm, 2),
                "height_cm": round(height_cm, 2),
                "area_cm2": round(area_cm2, 2),
                "perimeter_cm": round(2 * (width_cm + height_cm), 2)
            }
            
        except Exception:
            return None
    
    def _assess_clinical_severity(self, detection: ClinicalDetection) -> str:
        """Evaluar severidad clínica."""
        if not detection.detected:
            return "none"
        
        if not detection.lpp_grade:
            return "unknown"
        
        severity_map = {
            LPPGrade.GRADE_1: "mild",
            LPPGrade.GRADE_2: "moderate",
            LPPGrade.GRADE_3: "severe",
            LPPGrade.GRADE_4: "critical",
            LPPGrade.UNSTAGEABLE: "severe",
            LPPGrade.DEEP_TISSUE: "severe"
        }
        
        return severity_map.get(detection.lpp_grade, "unknown")
    
    def _determine_clinical_urgency(self, detection: ClinicalDetection) -> str:
        """Determinar urgencia clínica basada en detección."""
        if not detection.detected:
            return "routine"
        
        # Factores que aumentan urgencia
        if "infection_risk" in detection.risk_factors:
            return "urgent"
        
        if detection.lpp_grade in [LPPGrade.GRADE_3, LPPGrade.GRADE_4]:
            return "priority"
        
        if "tissue_necrosis" in detection.risk_factors:
            return "priority"
        
        return "routine"
    
    def _assess_detection_quality(self, detection: ClinicalDetection, quality_metrics: Dict[str, float]) -> float:
        """Evaluar calidad de la detección."""
        score = 0.0
        
        # Confianza de detección
        if detection.confidence >= 0.9:
            score += 0.4
        elif detection.confidence >= 0.8:
            score += 0.3
        elif detection.confidence >= 0.7:
            score += 0.2
        else:
            score += 0.1
        
        # Calidad de imagen
        image_quality = quality_metrics.get("overall_quality", 0)
        score += image_quality * 0.3
        
        # Completitud de features
        features_count = len([v for v in detection.clinical_features.values() if v])
        feature_completeness = features_count / len(detection.clinical_features) if detection.clinical_features else 0
        score += feature_completeness * 0.3
        
        return min(1.0, score)
    
    def _assess_documentation(self, detection: ClinicalDetection, metadata: Dict[str, Any]) -> float:
        """Evaluar completitud de documentación."""
        score = 0.0
        required_elements = 0
        present_elements = 0
        
        # Elementos requeridos
        requirements = [
            detection.detected,
            detection.lpp_grade is not None,
            len(detection.clinical_features) > 0,
            len(detection.risk_factors) > 0,
            len(detection.recommended_interventions) > 0,
            metadata.get("quality_metrics") is not None
        ]
        
        required_elements = len(requirements)
        present_elements = sum(requirements)
        
        return present_elements / required_elements if required_elements > 0 else 0.0
    
    def _determine_follow_up_needs(self, detection: ClinicalDetection) -> Dict[str, Any]:
        """Determinar necesidades de seguimiento."""
        follow_up = {
            "required": False,
            "timeline": "routine",
            "actions": []
        }
        
        if not detection.detected:
            return follow_up
        
        # Seguimiento basado en grado
        if detection.lpp_grade in [LPPGrade.GRADE_3, LPPGrade.GRADE_4]:
            follow_up["required"] = True
            follow_up["timeline"] = "24_hours"
            follow_up["actions"] = ["reassess_wound", "document_progress", "adjust_treatment"]
        elif detection.lpp_grade == LPPGrade.GRADE_2:
            follow_up["required"] = True
            follow_up["timeline"] = "48_hours"
            follow_up["actions"] = ["monitor_healing", "assess_interventions"]
        elif detection.lpp_grade == LPPGrade.GRADE_1:
            follow_up["required"] = True
            follow_up["timeline"] = "72_hours"
            follow_up["actions"] = ["verify_resolution", "continue_prevention"]
        
        # Ajustar por factores de riesgo
        if "infection_risk" in detection.risk_factors:
            follow_up["timeline"] = "12_hours"
            follow_up["actions"].append("infection_assessment")
        
        return follow_up
    
    def _generate_clinical_notes(self, analysis: Dict[str, Any]) -> str:
        """Generar notas clínicas estructuradas."""
        detection = analysis["detection"]
        
        notes = []
        notes.append(f"Clinical Assessment - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        notes.append("-" * 50)
        
        if detection.detected:
            notes.append(f"Pressure injury detected: Grade {detection.lpp_grade.value if detection.lpp_grade else 'Unknown'}")
            notes.append(f"Detection confidence: {detection.confidence:.2%}")
            notes.append(f"Clinical severity: {analysis['clinical_severity']}")
            notes.append(f"Urgency level: {analysis['urgency_level']}")
            
            if detection.measurement_data:
                notes.append(f"\nMeasurements:")
                notes.append(f"- Size: {detection.measurement_data['width_cm']} x {detection.measurement_data['height_cm']} cm")
                notes.append(f"- Area: {detection.measurement_data['area_cm2']} cm²")
            
            if detection.clinical_features:
                notes.append(f"\nClinical features:")
                for feature, value in detection.clinical_features.items():
                    notes.append(f"- {feature}: {value}")
            
            if detection.risk_factors:
                notes.append(f"\nRisk factors identified:")
                for factor in detection.risk_factors:
                    notes.append(f"- {factor}")
            
            if detection.recommended_interventions:
                notes.append(f"\nRecommended interventions:")
                for intervention in detection.recommended_interventions:
                    notes.append(f"- {intervention}")
            
            follow_up = analysis["follow_up_required"]
            if follow_up["required"]:
                notes.append(f"\nFollow-up required: {follow_up['timeline']}")
                notes.append("Follow-up actions:")
                for action in follow_up["actions"]:
                    notes.append(f"- {action}")
        else:
            notes.append("No pressure injury detected in the provided image.")
            notes.append("Recommend continued monitoring and prevention protocols.")
        
        notes.append(f"\nQuality assessment score: {analysis['quality_assessment']:.2f}/1.0")
        notes.append(f"Documentation completeness: {analysis['documentation_completeness']:.2%}")
        
        return "\n".join(notes)
    
    def _check_compliance_requirements(self, 
                                     detection: ClinicalDetection,
                                     metadata: Dict[str, Any],
                                     clinical_notes: str) -> List[str]:
        """Verificar requisitos de compliance médico."""
        flags = []
        
        # HIPAA compliance
        if clinical_notes and len(clinical_notes) > 0:
            flags.append("clinical_documentation_present")
        
        # Trazabilidad
        if metadata.get("processing_applied"):
            flags.append("processing_traceable")
        
        # Calidad de detección
        if detection.confidence >= self.min_confidence_threshold:
            flags.append("confidence_threshold_met")
        
        # Intervenciones documentadas
        if detection.recommended_interventions:
            flags.append("interventions_documented")
        
        # Mediciones objetivas
        if detection.measurement_data:
            flags.append("objective_measurements_recorded")
        
        return flags
    
    def _serialize_clinical_report(self, report: ClinicalReport) -> Dict[str, Any]:
        """Serializar reporte clínico para transmisión (solo datos tokenizados - NO PHI)."""
        return {
            "session_id": report.session_id,
            "tokenized_patient": report.tokenized_patient.to_dict(),  # NO PHI
            "detection": {
                "detected": report.detection_result.detected,
                "lpp_grade": report.detection_result.lpp_grade.value if report.detection_result.lpp_grade else None,
                "confidence": report.detection_result.confidence,
                "clinical_features": report.detection_result.clinical_features,
                "risk_factors": report.detection_result.risk_factors,
                "interventions": report.detection_result.recommended_interventions,
                "measurements": report.detection_result.measurement_data
            },
            "processing_time": report.processing_time,
            "clinical_notes": report.clinical_notes,
            "quality_metrics": report.quality_metrics,
            "compliance_flags": report.compliance_flags,
            "timestamp": report.timestamp.isoformat(),
            "processor_version": report.processor_version,
            "phi_protected": True  # Confirma que no hay PHI
        }
    
    def _determine_next_steps(self, report: ClinicalReport) -> List[str]:
        """Determinar próximos pasos basados en el reporte."""
        next_steps = []
        
        if report.detection_result.detected:
            # Notificación al equipo médico
            next_steps.append("notify_medical_team")
            
            # Actualizar plan de cuidados
            next_steps.append("update_care_plan")
            
            # Programar seguimiento
            next_steps.append("schedule_follow_up")
            
            # Si es severo, escalamiento
            if report.detection_result.lpp_grade in [LPPGrade.GRADE_3, LPPGrade.GRADE_4]:
                next_steps.append("escalate_to_specialist")
        else:
            # Continuar monitoreo preventivo
            next_steps.append("continue_prevention_protocol")
        
        return next_steps
    
    def _determine_notifications(self, report: ClinicalReport) -> List[Dict[str, Any]]:
        """Determinar notificaciones requeridas."""
        notifications = []
        
        if report.detection_result.detected:
            # Notificación básica (usando alias tokenizado - NO PHI)
            notifications.append({
                "type": "clinical_detection",
                "priority": "high" if report.detection_result.lpp_grade in [LPPGrade.GRADE_3, LPPGrade.GRADE_4] else "medium",
                "recipients": ["attending_physician", "nursing_team"],
                "summary": f"LPP Grade {report.detection_result.lpp_grade.value if report.detection_result.lpp_grade else 'Unknown'} detected for patient {report.tokenized_patient.patient_alias}",  # Batman, no Bruce Wayne
                "token_id": report.tokenized_patient.token_id,
                "phi_protected": True
            })
            
            # Notificación de riesgo si aplica (usando alias tokenizado - NO PHI)
            if "infection_risk" in report.detection_result.risk_factors:
                notifications.append({
                    "type": "infection_risk_alert",
                    "priority": "urgent",
                    "recipients": ["infection_control", "attending_physician"],
                    "summary": f"Infection risk identified for patient {report.tokenized_patient.patient_alias}",  # Batman, no Bruce Wayne
                    "token_id": report.tokenized_patient.token_id,
                    "phi_protected": True
                })
        
        return notifications
    
    def _get_recovery_suggestions(self, status: ProcessingStatus, error: str) -> List[str]:
        """Obtener sugerencias de recuperación basadas en el error."""
        suggestions = []
        
        if status == ProcessingStatus.VALIDATING:
            suggestions.extend([
                "Verify patient code format (XX-YYYY-NNN)",
                "Ensure image is attached to message",
                "Check image format is JPEG/PNG"
            ])
        elif status == ProcessingStatus.PREPROCESSING:
            suggestions.extend([
                "Try with a higher quality image",
                "Ensure good lighting in the image",
                "Avoid blurry or out-of-focus images"
            ])
        elif status == ProcessingStatus.DETECTING:
            suggestions.extend([
                "Ensure wound area is clearly visible",
                "Remove any obstructions from the image",
                "Try capturing from a different angle"
            ])
        else:
            suggestions.append("Contact technical support if issue persists")
        
        return suggestions


class ClinicalProcessingFactory:
    """Factory para crear instancias del sistema de procesamiento clínico."""
    
    @staticmethod
    async def create_processor(config: Optional[Dict[str, Any]] = None) -> ClinicalProcessingSystem:
        """Crear procesador clínico con configuración."""
        config = config or {}
        
        # Crear componentes
        detector = LPPDetector() if config.get("use_detector", True) else None
        preprocessor = ImagePreprocessor() if config.get("use_preprocessor", True) else None
        
        # Crear cliente de DB si está configurado
        db_client = None
        if config.get("use_database", False):
            db_client = SupabaseClient()
            await db_client.initialize()
        
        processor = ClinicalProcessingSystem(
            detector=detector,
            preprocessor=preprocessor,
            db_client=db_client
        )
        
        return processor