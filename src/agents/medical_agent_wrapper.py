"""
Medical Agent Wrapper for Testing
=================================

Wraps the ADK-based LPP medical agent functionality into a testable class
that can be used for medical testing without requiring ADK runtime.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from vigia_detect.systems.medical_decision_engine import make_evidence_based_decision


class LPPMedicalAgent:
    """
    Wrapper class for LPP Medical Agent functionality.
    Provides medical decision-making logic for testing purposes.
    """
    
    def __init__(self):
        """Initialize the LPP Medical Agent wrapper"""
        self.agent_name = "LPP_Medical_Agent"
        self.version = "1.0.0"
        self.medgemma_client = None  # Will be mocked in tests
        self.medical_knowledge = None  # Will be mocked in tests
        
        # LPP grade mappings based on NPUAP/EPUAP guidelines
        self.lpp_grade_descriptions = {
            0: "Sin evidencia de LPP",
            1: "LPP Grado I - Eritema no blanqueable",
            2: "LPP Grado II - Pérdida parcial del espesor de la piel",
            3: "LPP Grado III - Pérdida completa del espesor de la piel",
            4: "LPP Grado IV - Pérdida completa del tejido",
            5: "LPP No clasificable - Profundidad desconocida",
            6: "Sospecha de lesión de tejido profundo"
        }
        
        # Severity assessment mapping
        self.severity_mapping = {
            0: "RUTINA_PREVENTIVA",
            1: "ATENCIÓN",
            2: "IMPORTANTE", 
            3: "URGENTE",
            4: "EMERGENCY",
            5: "EVALUACIÓN_ESPECIALIZADA",
            6: "MONITOREO_ESTRICTO"
        }
    
    def analyze_lpp_image(self, image_path: str, token_id: str, 
                         detection_result: Dict[str, Any],
                         patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze LPP image and provide medical assessment.
        
        Args:
            image_path: Path to the medical image
            token_id: Batman token identifier (HIPAA compliant)
            detection_result: CV detection result
            patient_context: Additional patient medical context
            
        Returns:
            Complete medical analysis with recommendations
        """
        try:
            # Extract detection information
            detections = detection_result.get('detections', [])
            
            if not detections:
                # No LPP detected
                analysis = self._generate_no_lpp_analysis(token_id, patient_context)
                return {
                    'success': True,
                    'analysis': analysis,
                    'token_id': token_id,  # Batman token (HIPAA compliant)
                    'image_path': image_path,
                    'processing_timestamp': datetime.now().isoformat()
                }
            
            # Process detected LPP
            primary_detection = detections[0]  # Use highest confidence detection
            lpp_grade = self._extract_lpp_grade(primary_detection)
            confidence = primary_detection.get('confidence', 0.0)
            location = primary_detection.get('anatomical_location', 'unknown')
            
            # Generate evidence-based medical analysis
            evidence_based_analysis = make_evidence_based_decision(
                lpp_grade, confidence, location, patient_context
            )
            
            # Merge with legacy format for compatibility
            analysis = self._merge_with_legacy_format(evidence_based_analysis, token_id)
            
            return {
                'success': True,
                'analysis': analysis,
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'image_path': image_path,
                'processing_timestamp': datetime.now().isoformat(),
                'evidence_based_decision': evidence_based_analysis  # Nueva documentación médica
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'analysis': self._generate_error_analysis(str(e))
            }
    
    def _extract_lpp_grade(self, detection: Dict[str, Any]) -> int:
        """Extract LPP grade from detection class name"""
        class_name = detection.get('class', '')
        
        # Map detection classes to grades
        grade_mapping = {
            'lpp_grade_1': 1,
            'lpp_grade_2': 2,
            'lpp_grade_3': 3,
            'lpp_grade_4': 4,
            'lpp_unstageable': 5,
            'lpp_suspected_dti': 6,
            'no_lpp': 0
        }
        
        return grade_mapping.get(class_name, 0)
    
    def _generate_no_lpp_analysis(self, token_id: str, 
                                 patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis for cases with no LPP detected"""
        return {
            'lpp_grade': 0,
            'severity_assessment': 'RUTINA_PREVENTIVA',
            'confidence_score': 0.85,
            'anatomical_location': None,
            'clinical_recommendations': [
                'Continuar medidas preventivas según protocolo',
                'Evaluación Braden Scale regular',
                'Reposicionamiento cada 2 horas',
                'Mantener higiene e hidratación cutánea'
            ],
            'medical_warnings': [],
            'requires_human_review': False,
            'prevention_focused': True
        }
    
    def _generate_lpp_analysis(self, grade: int, confidence: float, location: str,
                              token_id: str, patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive LPP analysis"""
        
        analysis = {
            'lpp_grade': grade,
            'severity_assessment': self.severity_mapping.get(grade, 'EVALUACIÓN_REQUERIDA'),
            'confidence_score': confidence,
            'anatomical_location': location,
            'clinical_recommendations': self._get_grade_specific_recommendations(grade, location),
            'medical_warnings': [],
            'contraindications': [],
            'requires_human_review': confidence < 0.6 or grade >= 4,
            'requires_specialist_review': grade >= 3,
            'prevention_focused': False
        }
        
        # Add patient-specific considerations
        if patient_context:
            self._add_patient_specific_analysis(analysis, patient_context)
        
        # Add confidence-based adjustments
        self._add_confidence_adjustments(analysis, confidence)
        
        return analysis
    
    def _get_grade_specific_recommendations(self, grade: int, location: str) -> List[str]:
        """Get clinical recommendations specific to LPP grade and location"""
        
        base_recommendations = {
            1: [
                'Alivio inmediato de presión en área afectada',
                'Protección cutánea con film transparente',
                'Reposicionamiento cada 2 horas',
                'Evaluación de factores de riesgo'
            ],
            2: [
                'Curación húmeda con apósitos hidrocoloides',
                'Alivio total de presión en área',
                'Evaluación del dolor',
                'Documentación fotográfica semanal'
            ],
            3: [
                'Desbridamiento si tejido necrótico presente',
                'Apósitos avanzados según exudado',
                'Evaluación nutricional completa',
                'Consulta especializada en heridas'
            ],
            4: [
                'Evaluación quirúrgica urgente',
                'Manejo avanzado del dolor',
                'Cuidados multidisciplinarios',
                'Posible hospitalización'
            ],
            5: [
                'Evaluación especializada para clasificación',
                'Desbridamiento para determinar profundidad',
                'Monitoreo estricto de evolución'
            ],
            6: [
                'Monitoreo estricto evolución 24-48h',
                'Protección área sospechosa',
                'No masajes en área afectada',
                'Reevaluación especializada'
            ]
        }
        
        recommendations = base_recommendations.get(grade, ['Evaluación médica requerida'])
        
        # Add location-specific recommendations
        if location == 'heel':
            recommendations.append('Dispositivos de alivio de presión en talones')
            recommendations.append('Elevar talones de la superficie')
        elif location == 'sacrum':
            recommendations.append('Colchón de redistribución de presión')
            recommendations.append('Evitar cabecera >30° por periodos prolongados')
        elif location in ['elbow', 'shoulder']:
            recommendations.append('Protección con almohadillas blandas')
            recommendations.append('Cambios de posición frecuentes')
        
        return recommendations
    
    def _add_patient_specific_analysis(self, analysis: Dict[str, Any], 
                                     patient_context: Dict[str, Any]):
        """Add patient-specific medical considerations"""
        
        warnings = analysis['medical_warnings']
        contraindications = analysis['contraindications']
        recommendations = analysis['clinical_recommendations']
        
        # Diabetes considerations
        if patient_context.get('diabetes', False):
            warnings.append('Cicatrización retardada por diabetes')
            recommendations.append('Control glucémico estricto')
            recommendations.append('Inspección diaria de área')
        
        # Anticoagulant considerations
        if patient_context.get('anticoagulants', False):
            warnings.append('Riesgo de sangrado por anticoagulantes')
            contraindications.append('Evitar desbridamiento agresivo')
            recommendations.append('Precaución con manipulación')
        
        # Malnutrition considerations
        if patient_context.get('malnutrition', False):
            warnings.append('Cicatrización comprometida por malnutrición')
            recommendations.append('Evaluación nutricional urgente')
            recommendations.append('Suplementación proteica según indicación')
        
        # Circulation considerations
        if patient_context.get('compromised_circulation', False):
            warnings.append('Perfusión comprometida afecta cicatrización')
            recommendations.append('Evaluación vascular')
            contraindications.append('Contraindicado vendaje compresivo')
        
        # Immunosuppression considerations
        if patient_context.get('immunosuppression', False):
            warnings.append('Riesgo infeccioso aumentado por inmunosupresión')
            recommendations.append('Profilaxis antibiótica según protocolo')
            recommendations.append('Monitoreo signos de infección')
    
    def _add_confidence_adjustments(self, analysis: Dict[str, Any], confidence: float):
        """Add confidence-based adjustments to analysis"""
        
        if confidence < 0.5:
            analysis['requires_human_review'] = True
            analysis['human_review_reason'] = 'very_low_confidence_detection'
            analysis['severity_assessment'] += '_REQUIERE_VALIDACIÓN'
        elif confidence < 0.6:
            analysis['requires_human_review'] = True
            analysis['human_review_reason'] = 'low_confidence_detection'
            analysis['clinical_recommendations'].append('Validación por especialista recomendada')
        elif confidence < 0.7:
            analysis['clinical_recommendations'].append('Considerar segunda evaluación')
    
    def _merge_with_legacy_format(self, evidence_based_analysis: Dict[str, Any], token_id: str) -> Dict[str, Any]:
        """
        Fusiona el análisis basado en evidencia con el formato legacy para compatibilidad.
        """
        # Extraer campos del análisis basado en evidencia
        escalation = evidence_based_analysis.get('escalation_requirements', {})
        
        # Crear análisis en formato legacy compatible
        legacy_analysis = {
            'lpp_grade': evidence_based_analysis['lpp_grade'],
            'severity_assessment': evidence_based_analysis['severity_assessment'],
            'confidence_score': evidence_based_analysis['confidence_score'],
            'anatomical_location': evidence_based_analysis['anatomical_location'],
            'clinical_recommendations': evidence_based_analysis['clinical_recommendations'],
            'medical_warnings': evidence_based_analysis['medical_warnings'],
            'requires_human_review': escalation.get('requires_human_review', False),
            'requires_specialist_review': escalation.get('requires_specialist_review', False),
            'prevention_focused': evidence_based_analysis['lpp_grade'] == 0,
            # Nuevos campos con documentación médica
            'evidence_documentation': evidence_based_analysis.get('evidence_documentation', {}),
            'quality_metrics': evidence_based_analysis.get('quality_metrics', {}),
            'escalation_requirements': escalation
        }
        
        # Agregar campos específicos según escalación
        if escalation.get('escalation_reasons'):
            legacy_analysis['human_review_reason'] = escalation['escalation_reasons'][0]
            
        return legacy_analysis

    def _generate_error_analysis(self, error_message: str) -> Dict[str, Any]:
        """Generate analysis for error cases"""
        return {
            'lpp_grade': 0,
            'severity_assessment': 'ERROR_PROCESAMIENTO',
            'confidence_score': 0.0,
            'anatomical_location': None,
            'clinical_recommendations': [
                'Error en procesamiento - repetir análisis',
                'Evaluación manual requerida',
                'Contactar soporte técnico'
            ],
            'medical_warnings': ['Error en sistema de análisis'],
            'requires_human_review': True,
            'human_review_reason': 'processing_error',
            'error_details': error_message
        }


# Export for backward compatibility
lpp_agent = LPPMedicalAgent()

def procesar_imagen_lpp(image_path: str, token_id: str, detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function wrapper for compatibility - Now uses Batman tokens"""
    return lpp_agent.analyze_lpp_image(image_path, token_id, detection_result)

def generar_reporte_lpp(analysis: Dict[str, Any]) -> str:
    """Generate LPP report from analysis"""
    if not analysis.get('success', False):
        return f"Error en análisis: {analysis.get('error', 'Error desconocido')}"
    
    data = analysis['analysis']
    grade = data['lpp_grade']
    severity = data['severity_assessment']
    confidence = data['confidence_score']
    
    report = f"""
REPORTE DE ANÁLISIS LPP
======================
Token ID: {analysis.get('token_id', 'N/A')}  # Batman token (HIPAA compliant)
Fecha: {analysis.get('processing_timestamp', 'N/A')}

RESULTADO:
- Grado LPP: {grade}
- Severidad: {severity}
- Confianza: {confidence:.1%}
- Localización: {data.get('anatomical_location', 'N/A')}

RECOMENDACIONES:
{chr(10).join(f'- {rec}' for rec in data.get('clinical_recommendations', []))}

ADVERTENCIAS MÉDICAS:
{chr(10).join(f'- {warn}' for warn in data.get('medical_warnings', [])) if data.get('medical_warnings') else '- Ninguna'}

REVISIÓN REQUERIDA: {'Sí' if data.get('requires_human_review', False) else 'No'}
"""
    return report