"""
Enhanced Medical Agent Wrapper with MINSAL Integration
=====================================================

Enhanced wrapper that can use either international or MINSAL decision engines
based on configuration and patient context.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from vigia_detect.systems.medical_decision_engine import make_evidence_based_decision
from vigia_detect.systems.minsal_medical_decision_engine import make_minsal_clinical_decision


class EnhancedLPPMedicalAgent:
    """
    Enhanced LPP Medical Agent wrapper with MINSAL integration.
    Automatically selects appropriate decision engine based on context.
    """
    
    def __init__(self, use_minsal: bool = True, jurisdiction: str = "chile"):
        """
        Initialize enhanced medical agent.
        
        Args:
            use_minsal: Whether to use MINSAL integration
            jurisdiction: Medical jurisdiction (chile, international)
        """
        self.agent_name = "Enhanced_LPP_Medical_Agent"
        self.version = "1.1.0"
        self.use_minsal = use_minsal
        self.jurisdiction = jurisdiction.lower()
        
        # Decision engine selection
        self.decision_engine = self._select_decision_engine()
    
    def _select_decision_engine(self):
        """Select appropriate decision engine based on configuration"""
        if self.use_minsal and self.jurisdiction == "chile":
            return "minsal"
        else:
            return "international"
    
    def analyze_lpp_image(self, image_path: str, patient_code: str, 
                         detection_result: Dict[str, Any] = None,
                         patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced LPP analysis with automatic decision engine selection.
        
        Args:
            image_path: Path to the medical image
            patient_code: Unique patient identifier
            detection_result: CV pipeline detection results
            patient_context: Patient medical context
            
        Returns:
            Enhanced analysis with appropriate decision engine
        """
        try:
            # Extract detection information
            if detection_result and 'detections' in detection_result:
                detection = detection_result['detections'][0]
                lpp_grade = self._extract_lpp_grade(detection['class'])
                confidence = detection['confidence']
                anatomical_location = detection.get('anatomical_location', 'unknown')
            else:
                # Fallback for no detection
                lpp_grade = 0
                confidence = 0.9
                anatomical_location = 'general'
            
            # Select decision engine and make clinical decision
            if self.decision_engine == "minsal":
                # Use MINSAL enhanced decision engine
                clinical_decision = make_minsal_clinical_decision(
                    lpp_grade=lpp_grade,
                    confidence=confidence,
                    anatomical_location=anatomical_location,
                    patient_context=patient_context
                )
                
                # Extract MINSAL-specific information
                analysis = {
                    'lpp_grade': lpp_grade,
                    'lpp_description': clinical_decision.get('minsal_classification', ''),
                    'severity_assessment': clinical_decision['severity_assessment'],
                    'confidence_score': confidence,
                    'clinical_recommendations': clinical_decision['clinical_recommendations'],
                    'medical_warnings': clinical_decision.get('medical_warnings', []),
                    'prevention_measures': clinical_decision.get('prevention_measures', []),
                    'escalation_requirements': clinical_decision['escalation_requirements'],
                    'chilean_terminology': clinical_decision.get('chilean_terminology', {}),
                    'regulatory_compliance': clinical_decision.get('regulatory_compliance', {}),
                    'decision_engine': 'MINSAL_Enhanced',
                    'evidence_documentation': clinical_decision.get('evidence_documentation', {})
                }
                
            else:
                # Use international decision engine
                clinical_decision = make_evidence_based_decision(
                    lpp_grade=lpp_grade,
                    confidence=confidence,
                    anatomical_location=anatomical_location,
                    patient_context=patient_context
                )
                
                analysis = {
                    'lpp_grade': lpp_grade,
                    'lpp_description': self._get_lpp_description(lpp_grade),
                    'severity_assessment': clinical_decision['severity_assessment'],
                    'confidence_score': confidence,
                    'clinical_recommendations': clinical_decision['clinical_recommendations'],
                    'medical_warnings': clinical_decision.get('medical_warnings', []),
                    'escalation_requirements': clinical_decision['escalation_requirements'],
                    'decision_engine': 'International_NPUAP_EPUAP',
                    'evidence_documentation': clinical_decision.get('evidence_documentation', {})
                }
            
            return {
                'success': True,
                'analysis': analysis,
                'patient_code': patient_code,
                'image_path': image_path,
                'processing_timestamp': datetime.now().isoformat(),
                'agent_version': self.version,
                'jurisdiction': self.jurisdiction
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'patient_code': patient_code,
                'processing_timestamp': datetime.now().isoformat(),
                'agent_version': self.version
            }
    
    def _extract_lpp_grade(self, class_name: str) -> int:
        """Extract LPP grade from detection class name"""
        grade_mapping = {
            'lpp_grade_0': 0, 'no_lpp': 0,
            'lpp_grade_1': 1, 'lpp_stage_1': 1,
            'lpp_grade_2': 2, 'lpp_stage_2': 2,
            'lpp_grade_3': 3, 'lpp_stage_3': 3,
            'lpp_grade_4': 4, 'lpp_stage_4': 4,
            'lpp_unstageable': 5,
            'lpp_dti': 6, 'suspected_dti': 6
        }
        return grade_mapping.get(class_name.lower(), 0)
    
    def _get_lpp_description(self, grade: int) -> str:
        """Get LPP description for international classification"""
        descriptions = {
            0: "Sin evidencia de LPP",
            1: "LPP Grado I - Eritema no blanqueable",
            2: "LPP Grado II - Pérdida parcial del espesor",
            3: "LPP Grado III - Pérdida completa del espesor",
            4: "LPP Grado IV - Pérdida completa del tejido",
            5: "LPP No clasificable",
            6: "Sospecha de lesión de tejido profundo"
        }
        return descriptions.get(grade, "Clasificación desconocida")


# Factory function for easy instantiation
def create_medical_agent(jurisdiction: str = "chile", use_minsal: bool = True):
    """
    Factory function to create appropriate medical agent.
    
    Args:
        jurisdiction: Target jurisdiction (chile, international)
        use_minsal: Whether to use MINSAL integration for Chilean context
    
    Returns:
        Configured medical agent instance
    """
    if jurisdiction.lower() == "chile" and use_minsal:
        return EnhancedLPPMedicalAgent(use_minsal=True, jurisdiction="chile")
    else:
        return EnhancedLPPMedicalAgent(use_minsal=False, jurisdiction="international")