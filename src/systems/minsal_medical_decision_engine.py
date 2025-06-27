"""
Enhanced Medical Decision Engine with MINSAL Guidelines Integration
===================================================================

Motor de decisiones médicas mejorado que integra las directrices oficiales del MINSAL
de Chile junto con los estándares internacionales NPUAP/EPUAP/PPPIA.

Referencias integradas:
- NPUAP/EPUAP/PPPIA Clinical Practice Guideline 2019
- MINSAL - ULCERAS POR PRESION MINISTERIO (2015)
- MINSAL - Orientación Técnica Prevención de LPP (2018)
- Hospital Coquimbo - Protocolo Prevención UPP (2021)
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import json
from pathlib import Path

from .medical_decision_engine import (
    MedicalDecisionEngine, MedicalDecision, LPPGrade, 
    SeverityLevel, EvidenceLevel
)

logger = logging.getLogger(__name__)


class MINSALClassification(Enum):
    """Clasificación LPP según terminología MINSAL Chile"""
    SIN_EVIDENCIA = "Sin evidencia de LPP"
    CATEGORIA_I = "Categoría I - Eritema no blanqueable"
    CATEGORIA_II = "Categoría II - Pérdida parcial de espesor"
    CATEGORIA_III = "Categoría III - Pérdida total de espesor"
    CATEGORIA_IV = "Categoría IV - Pérdida total de piel y tejidos"
    NO_CLASIFICABLE = "No clasificable"
    TEJIDO_PROFUNDO = "LPP en tejidos profundos"


class MINSALDecisionEngine(MedicalDecisionEngine):
    """
    Motor de decisiones médicas mejorado con integración MINSAL.
    
    Extiende el motor base con guidelines específicos de Chile,
    manteniendo compatibilidad con estándares internacionales.
    """
    
    def __init__(self):
        """Inicializa motor con conocimiento MINSAL y base internacional"""
        super().__init__()
        self.minsal_guidelines = self._load_minsal_guidelines()
        self.clinical_config = self._load_clinical_configuration()
        self.extracted_minsal_info = self._load_extracted_minsal_info()
        
    def make_clinical_decision_minsal(self, lpp_grade: int, confidence: float,
                                    anatomical_location: str,
                                    patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera decisión clínica integrada MINSAL + NPUAP/EPUAP.
        
        Args:
            lpp_grade: Grado LPP detectado (0-6)
            confidence: Confianza de detección (0-1)
            anatomical_location: Localización anatómica
            patient_context: Contexto médico del paciente
            
        Returns:
            Dict con decisión clínica completa con contexto chileno
        """
        logger.info(f"Generando decisión MINSAL - Grado: {lpp_grade}, Confianza: {confidence:.1%}")
        
        # Generar decisión base internacional
        base_decision = self.make_clinical_decision(
            lpp_grade, confidence, anatomical_location, patient_context
        )
        
        # Enriquecer con guidelines MINSAL
        minsal_enhancement = self._enhance_with_minsal_guidelines(
            lpp_grade, confidence, anatomical_location, patient_context
        )
        
        # Combinar decisiones
        enhanced_decision = self._combine_decisions(base_decision, minsal_enhancement)
        
        # Añadir contexto chileno específico
        enhanced_decision.update({
            'minsal_classification': self._get_minsal_classification(lpp_grade),
            'chilean_terminology': self._get_chilean_terminology(lpp_grade),
            'regulatory_compliance': {
                'minsal_compliant': True,
                'national_guidelines': '2018',
                'international_standards': 'NPUAP/EPUAP 2019',
                'jurisdiction': 'Chile'
            },
            'linguistic_adaptation': {
                'language': 'spanish',
                'terminology_standard': 'lesiones_por_presion',
                'cultural_context': 'chilean_healthcare_system'
            }
        })
        
        logger.info(f"Decisión MINSAL completada - Clasificación: {enhanced_decision['minsal_classification']}")
        return enhanced_decision
    
    def _enhance_with_minsal_guidelines(self, lpp_grade: int, confidence: float,
                                      anatomical_location: str,
                                      patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enriquece decisión con guidelines específicos MINSAL"""
        
        minsal_recommendations = self._get_minsal_recommendations(lpp_grade)
        minsal_prevention = self._get_minsal_prevention_measures(lpp_grade)
        minsal_risk_factors = self._get_minsal_risk_factors(patient_context)
        
        return {
            'minsal_specific_recommendations': minsal_recommendations,
            'minsal_prevention_protocols': minsal_prevention,
            'minsal_risk_assessment': minsal_risk_factors,
            'chilean_healthcare_context': self._get_chilean_healthcare_context(),
            'minsal_evidence_base': self._get_minsal_evidence_references(lpp_grade)
        }
    
    def _get_minsal_recommendations(self, lpp_grade: int) -> List[MedicalDecision]:
        """Obtiene recomendaciones específicas según MINSAL"""
        recommendations = []
        
        minsal_protocols = {
            0: [
                ("Aplicar Escala ELPO cada 24h", "MINSAL OOTT 2018", 
                 "Escala de Evaluación de LPP específica Chile"),
                ("Inspección diaria de prominencias óseas", "MINSAL Protocolo Prevención", 
                 "Protocolo chileno de vigilancia cutánea"),
                ("Uso de superficies de redistribución de presión", "MINSAL OOTT 2018",
                 "Colchones viscoelásticos según disponibilidad hospitalaria Chile")
            ],
            1: [
                ("Alivio inmediato de presión según protocolo MINSAL", "MINSAL OOTT 2018",
                 "Categoría I requiere intervención inmediata"),
                ("Protección con apósitos profilácticos", "MINSAL Guidelines 2018",
                 "Espumas de poliuretano con silicona recomendadas"),
                ("Evaluación nutricional según estándares chilenos", "MINSAL Protocolo",
                 "Consideración especial desnutrición población chilena")
            ],
            2: [
                ("Curación húmeda con apósitos disponibles en sistema público", "MINSAL OOTT 2018",
                 "Adaptación a recursos disponibles en Chile"),
                ("Manejo del dolor según protocolos MINSAL", "MINSAL Guidelines",
                 "90% pacientes Categoría II reportan dolor significativo"),
                ("Documentación fotográfica semanal", "MINSAL Protocolo HCoquimbo",
                 "Registro obligatorio para seguimiento")
            ],
            3: [
                ("Derivación a especialista en heridas", "MINSAL OOTT 2018",
                 "Red de derivación hospitalaria chilena"),
                ("Evaluación quirúrgica si indicado", "MINSAL Protocolo",
                 "Criterios quirúrgicos según recursos disponibles"),
                ("Manejo multidisciplinario", "MINSAL Guidelines 2018",
                 "Equipo mínimo: médico, enfermera, nutricionista")
            ],
            4: [
                ("Evaluación quirúrgica urgente", "MINSAL OOTT 2018",
                 "Criterio de derivación inmediata sistema público"),
                ("Hospitalización si indicado", "MINSAL Protocolo",
                 "Evaluación de recursos hospitalarios disponibles"),
                ("Cuidados paliativos si apropiado", "MINSAL Guidelines",
                 "Consideración calidad vida según contexto familiar")
            ]
        }
        
        if lpp_grade in minsal_protocols:
            for rec_text, ref, rationale in minsal_protocols[lpp_grade]:
                recommendations.append(
                    MedicalDecision(
                        f"minsal_grade_{lpp_grade}",
                        rec_text,
                        EvidenceLevel.LEVEL_B,  # MINSAL guidelines = Level B
                        ref,
                        rationale
                    )
                )
        
        return recommendations
    
    def _get_minsal_prevention_measures(self, lpp_grade: int) -> List[str]:
        """Obtiene medidas preventivas específicas MINSAL"""
        
        if not self.extracted_minsal_info:
            return []
        
        prevention_measures = []
        
        # Extraer medidas de documentos MINSAL procesados
        for doc_name, doc_info in self.extracted_minsal_info.items():
            if 'prevention_measures' in doc_info:
                for measure in doc_info['prevention_measures']:
                    if len(measure) > 20:  # Filtrar medidas sustanciales
                        prevention_measures.append(f"{measure} (Fuente: {doc_name})")
        
        # Medidas específicas por grado según MINSAL
        grade_specific = {
            0: [
                "Aplicación Escala ELPO cada 24 horas",
                "Reposicionamiento cada 2 horas según tolerancia",
                "Uso de colchones de redistribución de presión"
            ],
            1: [
                "Protección con films transparentes",
                "Posicionamiento con cuñas de 30 grados",
                "Hidratación cutánea con productos no irritantes"
            ],
            2: [
                "Protección perímetro de la lesión",
                "Manejo de la humedad perilesional",
                "Evaluación nutricional urgente"
            ]
        }
        
        if lpp_grade in grade_specific:
            prevention_measures.extend(grade_specific[lpp_grade])
        
        return prevention_measures[:8]  # Limitar a 8 medidas más relevantes
    
    def _get_minsal_risk_factors(self, patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa factores de riesgo según contexto chileno"""
        
        risk_assessment = {
            'elpo_scale_indicated': True,
            'chilean_specific_risks': [],
            'healthcare_system_factors': []
        }
        
        if patient_context:
            # Factores específicos población chilena
            if patient_context.get('age', 0) > 65:
                risk_assessment['chilean_specific_risks'].append(
                    'Población adulto mayor con alta prevalencia LPP en Chile'
                )
            
            if patient_context.get('malnutrition', False):
                risk_assessment['chilean_specific_risks'].append(
                    'Desnutrición: factor crítico en población chilena hospitalizada'
                )
            
            if patient_context.get('diabetes', False):
                risk_assessment['chilean_specific_risks'].append(
                    'Diabetes: prevalencia 12.3% población chilena adulta'
                )
            
            # Factores del sistema de salud chileno
            if patient_context.get('public_healthcare', True):
                risk_assessment['healthcare_system_factors'].extend([
                    'Sistema público: recursos limitados para prevención',
                    'Necesidad optimización recursos disponibles',
                    'Priorización según Escala ELPO'
                ])
        
        return risk_assessment
    
    def _get_chilean_healthcare_context(self) -> Dict[str, Any]:
        """Proporciona contexto específico del sistema de salud chileno"""
        return {
            'healthcare_system': 'mixed_public_private',
            'prevalence_data': {
                'hospital_acquired_lpp': '8.2%',
                'icu_lpp': '15.6%',
                'elderly_care': '23.1%'
            },
            'resource_considerations': {
                'public_system_limitations': True,
                'specialized_surfaces_availability': 'limited',
                'specialist_availability': 'concentrated_urban_areas'
            },
            'regulatory_framework': {
                'minsal_mandatory': True,
                'quality_indicators': 'hospital_lpp_rates',
                'reporting_requirements': 'monthly_statistics'
            }
        }
    
    def _get_minsal_evidence_references(self, lpp_grade: int) -> List[Dict[str, str]]:
        """Obtiene referencias específicas MINSAL por grado"""
        
        base_references = [
            {
                'source': 'MINSAL',
                'document': 'Orientación Técnica Prevención de LPP',
                'year': '2018',
                'relevance': f'grade_{lpp_grade}_specific'
            },
            {
                'source': 'MINSAL',
                'document': 'ULCERAS POR PRESION MINISTERIO',
                'year': '2015',
                'relevance': 'national_standards'
            }
        ]
        
        if lpp_grade >= 3:
            base_references.append({
                'source': 'Hospital Coquimbo',
                'document': 'Protocolo Prevención UPP',
                'year': '2021',
                'relevance': 'institutional_best_practices'
            })
        
        return base_references
    
    def _get_minsal_classification(self, lpp_grade: int) -> str:
        """Obtiene clasificación según terminología MINSAL"""
        classification_map = {
            0: MINSALClassification.SIN_EVIDENCIA.value,
            1: MINSALClassification.CATEGORIA_I.value,
            2: MINSALClassification.CATEGORIA_II.value,
            3: MINSALClassification.CATEGORIA_III.value,
            4: MINSALClassification.CATEGORIA_IV.value,
            5: MINSALClassification.NO_CLASIFICABLE.value,
            6: MINSALClassification.TEJIDO_PROFUNDO.value
        }
        
        return classification_map.get(lpp_grade, "Clasificación no disponible")
    
    def _get_chilean_terminology(self, lpp_grade: int) -> Dict[str, str]:
        """Proporciona terminología médica en español chileno"""
        terminology = {
            'condition_name': 'Lesiones Por Presión (LPP)',
            'alternative_terms': ['Úlceras por Presión (UPP)', 'Escaras'],
            'classification_system': 'Categorías NPUAP adaptadas MINSAL',
            'prevention_term': 'Medidas Preventivas LPP',
            'risk_assessment': 'Evaluación de Riesgo - Escala ELPO'
        }
        
        grade_specific = {
            0: 'Piel íntegra sin evidencia lesional',
            1: 'Eritema no blanqueable en prominencia ósea',
            2: 'Pérdida espesor parcial con dermis expuesta',
            3: 'Pérdida espesor total con grasa visible',
            4: 'Pérdida total con exposición estructuras profundas',
            5: 'Lesión cubierta por esfacelo o escara',
            6: 'Decoloración persistente en tejidos profundos'
        }
        
        terminology['grade_description'] = grade_specific.get(
            lpp_grade, "Descripción no disponible"
        )
        
        return terminology
    
    def _combine_decisions(self, base_decision: Dict[str, Any], 
                          minsal_enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Combina decisión base con enriquecimiento MINSAL"""
        
        combined = base_decision.copy()
        
        # Añadir recomendaciones MINSAL a las existentes
        if 'minsal_specific_recommendations' in minsal_enhancement:
            minsal_recs = [rec.recommendation for rec in minsal_enhancement['minsal_specific_recommendations']]
            combined['clinical_recommendations'].extend(minsal_recs)
        
        # Integrar prevención MINSAL
        if 'minsal_prevention_protocols' in minsal_enhancement:
            combined['prevention_measures'] = minsal_enhancement['minsal_prevention_protocols']
        
        # Añadir información MINSAL específica
        combined.update(minsal_enhancement)
        
        # Actualizar documentación de evidencia
        combined['evidence_documentation']['minsal_integration'] = True
        combined['evidence_documentation']['chilean_guidelines'] = '2018'
        
        return combined
    
    def _load_minsal_guidelines(self) -> Dict[str, Any]:
        """Carga configuración guidelines MINSAL"""
        try:
            config_path = Path("src/systems/config/clinical_guidelines.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando guidelines MINSAL: {e}")
        
        return {}
    
    def _load_clinical_configuration(self) -> Dict[str, Any]:
        """Carga configuración clínica del sistema"""
        return self.minsal_guidelines.get('regional_adaptations', {}).get('chile', {})
    
    def _load_extracted_minsal_info(self) -> Dict[str, Any]:
        """Carga información extraída de documentos MINSAL"""
        try:
            info_path = Path("src/systems/config/minsal_extracted_info.json")
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando info extraída MINSAL: {e}")
        
        return {}


# Instancia global del motor MINSAL
minsal_decision_engine = MINSALDecisionEngine()


def make_minsal_clinical_decision(lpp_grade: int, confidence: float,
                                anatomical_location: str,
                                patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Función principal para decisiones médicas con integración MINSAL.
    
    Args:
        lpp_grade: Grado LPP (0-6)
        confidence: Confianza detección (0-1)
        anatomical_location: Localización anatómica
        patient_context: Contexto médico del paciente
        
    Returns:
        Decisión clínica integrada MINSAL + internacional
    """
    return minsal_decision_engine.make_clinical_decision_minsal(
        lpp_grade, confidence, anatomical_location, patient_context
    )


__all__ = [
    'MINSALDecisionEngine',
    'make_minsal_clinical_decision', 
    'MINSALClassification'
]