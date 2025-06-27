"""
MedGemma client for medical AI analysis in Vigia system.

Este cliente especializado integra Google MedGemma para análisis médico específico de LPP,
manteniendo compliance HIPAA y auditabilidad completa.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from src.core.base_client import BaseClient as BaseClientV2
from src.utils.secure_logger import SecureLogger
from src.utils.error_handling import handle_exceptions
from config.settings import get_settings

logger = SecureLogger(__name__)
settings = get_settings()


class MedicalAnalysisType(Enum):
    """Tipos de análisis médico disponibles"""
    LPP_GRADING = "lpp_grading"
    CLINICAL_TRIAGE = "clinical_triage"
    INTERVENTION_PLANNING = "intervention_planning"
    RISK_ASSESSMENT = "risk_assessment"
    CLINICAL_DOCUMENTATION = "clinical_documentation"


@dataclass
class MedicalContext:
    """Contexto médico para análisis MedGemma"""
    patient_age: Optional[int] = None
    medical_history: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    mobility_status: Optional[str] = None
    risk_factors: Optional[List[str]] = None
    previous_lpp_history: Optional[bool] = None


@dataclass
class MedGemmaResponse:
    """Respuesta estructurada de MedGemma"""
    analysis_type: MedicalAnalysisType
    confidence_score: float
    clinical_findings: Dict[str, Any]
    recommendations: List[str]
    risk_level: Optional[str] = None
    follow_up_needed: bool = False
    audit_trail: Dict[str, Any] = None


class MedGemmaClient(BaseClientV2):
    """
    Cliente especializado para MedGemma con enfoque en análisis médico LPP.
    
    Implementa:
    - Análisis clínico especializado en LPP
    - Generación de documentación médica
    - Evaluación de riesgo basada en evidencia
    - Compliance HIPAA con audit trail completo
    """

    def __init__(self):
        # Campos requeridos para MedGemma
        required_fields = []  # No hay campos estrictamente requeridos
        optional_fields = ["google_api_key", "medgemma_model", "google_cloud_project"]
        
        # Configuración MedGemma
        # Primero revisar variable de entorno, luego settings
        self.model_name = os.environ.get('MEDGEMMA_MODEL') or (settings.medgemma_model if hasattr(settings, 'medgemma_model') else "gemini-1.5-pro")
        self.project_id = settings.google_cloud_project if hasattr(settings, 'google_cloud_project') else None
        
        # Configuración de seguridad médica
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Configuración del modelo
        self.generation_config = {
            "temperature": 0.1,  # Baja creatividad para análisis médico
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        super().__init__(
            service_name="medgemma",
            required_fields=required_fields,
            optional_fields=optional_fields
        )

    def _initialize_client(self):
        """Inicializar cliente MedGemma"""
        try:
            # Primero revisar variable de entorno, luego settings
            api_key = os.environ.get('GOOGLE_API_KEY') or (settings.google_api_key if hasattr(settings, 'google_api_key') else None)
            
            if api_key and api_key != "placeholder_google_key":
                genai.configure(api_key=api_key)
                logger.info("Google API key configured successfully")
            else:
                logger.warning("Google API key not configured, using default authentication")
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            logger.info(f"MedGemma client initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MedGemma client: {e}")
            raise

    @handle_exceptions(default_return=None)
    async def analyze_lpp_findings(
        self,
        clinical_observations: str,
        visual_findings: Optional[Dict[str, Any]] = None,
        medical_context: Optional[MedicalContext] = None
    ) -> Optional[MedGemmaResponse]:
        """
        Analizar hallazgos clínicos de LPP usando MedGemma.
        
        Args:
            clinical_observations: Observaciones clínicas en texto
            visual_findings: Resultados del pipeline de visión (opcional)
            medical_context: Contexto médico del paciente (opcional)
            
        Returns:
            MedGemmaResponse con análisis médico estructurado
        """
        
        prompt = self._build_lpp_analysis_prompt(
            clinical_observations, visual_findings, medical_context
        )
        
        try:
            response = await self._generate_medical_response(
                prompt, MedicalAnalysisType.LPP_GRADING
            )
            
            return self._parse_lpp_response(response)
            
        except Exception as e:
            logger.error(f"Error in LPP analysis: {e}")
            return None

    @handle_exceptions(default_return=None)
    async def perform_clinical_triage(
        self,
        patient_message: str,
        symptoms_description: str,
        medical_context: Optional[MedicalContext] = None
    ) -> Optional[MedGemmaResponse]:
        """
        Realizar triage clínico usando MedGemma.
        
        Args:
            patient_message: Mensaje del paciente
            symptoms_description: Descripción de síntomas
            medical_context: Contexto médico (opcional)
            
        Returns:
            MedGemmaResponse con evaluación de triage
        """
        
        prompt = self._build_triage_prompt(patient_message, symptoms_description, medical_context)
        
        try:
            response = await self._generate_medical_response(
                prompt, MedicalAnalysisType.CLINICAL_TRIAGE
            )
            
            return self._parse_triage_response(response)
            
        except Exception as e:
            logger.error(f"Error in clinical triage: {e}")
            return None

    @handle_exceptions(default_return=None)
    async def generate_intervention_plan(
        self,
        lpp_grade: str,
        risk_factors: List[str],
        patient_limitations: Optional[List[str]] = None,
        medical_context: Optional[MedicalContext] = None
    ) -> Optional[MedGemmaResponse]:
        """
        Generar plan de intervención personalizado usando MedGemma.
        
        Args:
            lpp_grade: Grado de LPP detectado
            risk_factors: Factores de riesgo identificados
            patient_limitations: Limitaciones del paciente (opcional)
            medical_context: Contexto médico (opcional)
            
        Returns:
            MedGemmaResponse con plan de intervención
        """
        
        prompt = self._build_intervention_prompt(
            lpp_grade, risk_factors, patient_limitations, medical_context
        )
        
        try:
            response = await self._generate_medical_response(
                prompt, MedicalAnalysisType.INTERVENTION_PLANNING
            )
            
            return self._parse_intervention_response(response)
            
        except Exception as e:
            logger.error(f"Error in intervention planning: {e}")
            return None

    def _build_lpp_analysis_prompt(
        self,
        clinical_observations: str,
        visual_findings: Optional[Dict[str, Any]],
        medical_context: Optional[MedicalContext]
    ) -> str:
        """Construir prompt especializado para análisis de LPP"""
        
        prompt = f"""
Como especialista en medicina interna con expertise en lesiones por presión (LPP), analiza los siguientes hallazgos clínicos:

OBSERVACIONES CLÍNICAS:
{clinical_observations}

"""

        if visual_findings:
            prompt += f"""
HALLAZGOS VISUALES (Computer Vision):
- Detecciones: {visual_findings.get('detections', 'No disponible')}
- Características: {visual_findings.get('features', 'No disponible')}
- Confianza del modelo: {visual_findings.get('confidence', 'No disponible')}

"""

        if medical_context:
            prompt += f"""
CONTEXTO MÉDICO DEL PACIENTE:
- Edad: {medical_context.patient_age or 'No especificada'}
- Historia médica: {', '.join(medical_context.medical_history) if medical_context.medical_history else 'No disponible'}
- Medicamentos actuales: {', '.join(medical_context.current_medications) if medical_context.current_medications else 'No disponible'}
- Estado de movilidad: {medical_context.mobility_status or 'No especificado'}
- Factores de riesgo: {', '.join(medical_context.risk_factors) if medical_context.risk_factors else 'No identificados'}
- Historia previa de LPP: {'Sí' if medical_context.previous_lpp_history else 'No' if medical_context.previous_lpp_history is not None else 'No especificado'}

"""

        prompt += """
INSTRUCCIONES DE ANÁLISIS:
1. Evalúa el grado de LPP según clasificación NPUAP/EPUAP (Grado 1-4 o sospecha de lesión de tejido profundo)
2. Identifica factores de riesgo presentes
3. Evalúa la urgencia clínica (baja, moderada, alta, crítica)
4. Proporciona recomendaciones específicas basadas en evidencia
5. Indica si se requiere seguimiento médico inmediato

FORMATO DE RESPUESTA (JSON):
{
    "lpp_grade": "Grado específico o 'No LPP detectada'",
    "confidence_score": 0.85,
    "clinical_urgency": "moderada",
    "risk_factors_identified": ["factor1", "factor2"],
    "tissue_involvement": "descripción del compromiso tisular",
    "location_assessment": "descripción de la localización",
    "immediate_recommendations": ["recomendación1", "recomendación2"],
    "follow_up_needed": true,
    "clinical_rationale": "explicación del razonamiento clínico"
}
"""
        
        return prompt

    def _build_triage_prompt(
        self,
        patient_message: str,
        symptoms_description: str,
        medical_context: Optional[MedicalContext]
    ) -> str:
        """Construir prompt para triage clínico"""
        
        prompt = f"""
Como médico especialista en triage de urgencias con expertise en lesiones por presión, evalúa la siguiente consulta:

MENSAJE DEL PACIENTE:
{patient_message}

DESCRIPCIÓN DE SÍNTOMAS:
{symptoms_description}

"""

        if medical_context:
            prompt += f"""
CONTEXTO MÉDICO:
- Edad: {medical_context.patient_age or 'No especificada'}
- Historia médica relevante: {', '.join(medical_context.medical_history) if medical_context.medical_history else 'No disponible'}
- Estado de movilidad: {medical_context.mobility_status or 'No especificado'}
- Factores de riesgo conocidos: {', '.join(medical_context.risk_factors) if medical_context.risk_factors else 'No identificados'}

"""

        prompt += """
INSTRUCCIONES DE TRIAGE:
1. Evalúa la urgencia clínica según criterios de triage médico
2. Identifica signos de alarma o complicaciones
3. Determina el nivel de atención requerido
4. Proporciona recomendaciones inmediatas
5. Establece timeframe para evaluación médica

NIVELES DE URGENCIA:
- CRÍTICA: Requiere atención inmediata (< 30 min)
- ALTA: Requiere evaluación urgente (< 2 horas)  
- MODERADA: Requiere evaluación programada (< 24 horas)
- BAJA: Puede manejar con cuidados domiciliarios + seguimiento

FORMATO DE RESPUESTA (JSON):
{
    "urgency_level": "moderada",
    "confidence_score": 0.90,
    "alarm_signs": ["signo1", "signo2"],
    "recommended_care_level": "evaluación programada",
    "timeframe_hours": 24,
    "immediate_actions": ["acción1", "acción2"],
    "red_flags": ["bandera_roja1"] or [],
    "disposition": "descripción de la disposición recomendada",
    "clinical_reasoning": "explicación del razonamiento"
}
"""
        
        return prompt

    def _build_intervention_prompt(
        self,
        lpp_grade: str,
        risk_factors: List[str],
        patient_limitations: Optional[List[str]],
        medical_context: Optional[MedicalContext]
    ) -> str:
        """Construir prompt para planificación de intervenciones"""
        
        prompt = f"""
Como especialista en cuidado de heridas y prevención de LPP, desarrolla un plan de intervención personalizado:

DIAGNÓSTICO:
Lesión por Presión - {lpp_grade}

FACTORES DE RIESGO IDENTIFICADOS:
{', '.join(risk_factors)}

"""

        if patient_limitations:
            prompt += f"""
LIMITACIONES DEL PACIENTE:
{', '.join(patient_limitations)}

"""

        if medical_context:
            prompt += f"""
CONTEXTO CLÍNICO:
- Edad: {medical_context.patient_age or 'No especificada'}
- Estado de movilidad: {medical_context.mobility_status or 'No especificado'}
- Medicamentos: {', '.join(medical_context.current_medications) if medical_context.current_medications else 'No especificado'}

"""

        prompt += """
INSTRUCCIONES PARA PLAN DE INTERVENCIÓN:
1. Intervenciones específicas basadas en el grado de LPP
2. Estrategias de prevención de nuevas lesiones
3. Manejo del dolor y comfort del paciente
4. Educación al paciente/cuidador
5. Criterios de seguimiento y evaluación
6. Recursos necesarios y feasibilidad

FORMATO DE RESPUESTA (JSON):
{
    "intervention_priority": "alta",
    "confidence_score": 0.88,
    "primary_interventions": [
        {
            "intervention": "nombre",
            "frequency": "frecuencia",
            "duration": "duración",
            "instructions": "instrucciones específicas"
        }
    ],
    "prevention_strategies": ["estrategia1", "estrategia2"],
    "pain_management": ["método1", "método2"],
    "patient_education": ["tema1", "tema2"],
    "follow_up_schedule": "cronograma de seguimiento",
    "resources_needed": ["recurso1", "recurso2"],
    "success_criteria": ["criterio1", "criterio2"],
    "contraindications": ["contraindicación1"] or []
}
"""
        
        return prompt

    async def _generate_medical_response(
        self,
        prompt: str,
        analysis_type: MedicalAnalysisType
    ) -> str:
        """Generar respuesta médica usando MedGemma"""
        
        try:
            # Logging para auditoria (sin contenido PHI)
            logger.info(f"Generating medical response for: {analysis_type.value}")
            
            # Llamada al modelo
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from MedGemma")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating medical response: {e}")
            raise

    def _parse_lpp_response(self, response_text: str) -> MedGemmaResponse:
        """Parsear respuesta de análisis LPP"""
        try:
            # Extraer JSON de la respuesta
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_response = json.loads(json_str)
                
                return MedGemmaResponse(
                    analysis_type=MedicalAnalysisType.LPP_GRADING,
                    confidence_score=parsed_response.get('confidence_score', 0.0),
                    clinical_findings={
                        'lpp_grade': parsed_response.get('lpp_grade'),
                        'tissue_involvement': parsed_response.get('tissue_involvement'),
                        'location_assessment': parsed_response.get('location_assessment'),
                        'clinical_rationale': parsed_response.get('clinical_rationale')
                    },
                    recommendations=parsed_response.get('immediate_recommendations', []),
                    risk_level=parsed_response.get('clinical_urgency'),
                    follow_up_needed=parsed_response.get('follow_up_needed', False),
                    audit_trail={
                        'timestamp': datetime.now().isoformat(),
                        'model_used': self.model_name,
                        'risk_factors_identified': parsed_response.get('risk_factors_identified', [])
                    }
                )
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing LPP response: {e}")
            
        # Fallback response
        return MedGemmaResponse(
            analysis_type=MedicalAnalysisType.LPP_GRADING,
            confidence_score=0.0,
            clinical_findings={'error': 'Failed to parse response'},
            recommendations=['Require manual clinical review'],
            follow_up_needed=True
        )

    def _parse_triage_response(self, response_text: str) -> MedGemmaResponse:
        """Parsear respuesta de triage clínico"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_response = json.loads(json_str)
                
                return MedGemmaResponse(
                    analysis_type=MedicalAnalysisType.CLINICAL_TRIAGE,
                    confidence_score=parsed_response.get('confidence_score', 0.0),
                    clinical_findings={
                        'urgency_level': parsed_response.get('urgency_level'),
                        'alarm_signs': parsed_response.get('alarm_signs', []),
                        'red_flags': parsed_response.get('red_flags', []),
                        'disposition': parsed_response.get('disposition'),
                        'clinical_reasoning': parsed_response.get('clinical_reasoning')
                    },
                    recommendations=parsed_response.get('immediate_actions', []),
                    risk_level=parsed_response.get('urgency_level'),
                    follow_up_needed=parsed_response.get('timeframe_hours', 24) <= 24,
                    audit_trail={
                        'timestamp': datetime.now().isoformat(),
                        'model_used': self.model_name,
                        'recommended_care_level': parsed_response.get('recommended_care_level')
                    }
                )
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing triage response: {e}")
            
        # Fallback response
        return MedGemmaResponse(
            analysis_type=MedicalAnalysisType.CLINICAL_TRIAGE,
            confidence_score=0.0,
            clinical_findings={'error': 'Failed to parse response'},
            recommendations=['Require immediate medical evaluation'],
            risk_level='alta',
            follow_up_needed=True
        )

    def _parse_intervention_response(self, response_text: str) -> MedGemmaResponse:
        """Parsear respuesta de planificación de intervenciones"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_response = json.loads(json_str)
                
                return MedGemmaResponse(
                    analysis_type=MedicalAnalysisType.INTERVENTION_PLANNING,
                    confidence_score=parsed_response.get('confidence_score', 0.0),
                    clinical_findings={
                        'primary_interventions': parsed_response.get('primary_interventions', []),
                        'prevention_strategies': parsed_response.get('prevention_strategies', []),
                        'pain_management': parsed_response.get('pain_management', []),
                        'contraindications': parsed_response.get('contraindications', [])
                    },
                    recommendations=[
                        f"Follow-up: {parsed_response.get('follow_up_schedule', 'TBD')}",
                        f"Success criteria: {', '.join(parsed_response.get('success_criteria', []))}"
                    ],
                    risk_level=parsed_response.get('intervention_priority'),
                    follow_up_needed=True,
                    audit_trail={
                        'timestamp': datetime.now().isoformat(),
                        'model_used': self.model_name,
                        'resources_needed': parsed_response.get('resources_needed', [])
                    }
                )
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing intervention response: {e}")
            
        # Fallback response
        return MedGemmaResponse(
            analysis_type=MedicalAnalysisType.INTERVENTION_PLANNING,
            confidence_score=0.0,
            clinical_findings={'error': 'Failed to parse response'},
            recommendations=['Require clinical consultation for intervention planning'],
            follow_up_needed=True
        )

    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del cliente MedGemma"""
        try:
            test_prompt = "Respond with 'MedGemma health check OK' if you can process medical queries."
            response = await self._generate_medical_response(
                test_prompt, MedicalAnalysisType.CLINICAL_TRIAGE
            )
            
            return {
                'status': 'healthy' if 'OK' in response else 'degraded',
                'model': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'response_received': bool(response)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_connection(self) -> bool:
        """
        Implementación del método abstracto de BaseClientV2.
        Valida la conexión con el servicio MedGemma.
        
        Returns:
            bool: True si la conexión es válida, False en caso contrario
        """
        try:
            # Intentar generar contenido simple para validar la conexión
            response = await asyncio.to_thread(
                self.model.generate_content,
                "Test connection"
            )
            return bool(response.text)
        except Exception as e:
            logger.error(f"MedGemma connection validation failed: {e}")
            return False