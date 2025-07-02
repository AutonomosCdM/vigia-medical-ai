"""
Medical prompts and templates for AI-powered clinical analysis in Vigia.

Este módulo contiene prompts especializados para análisis médico, diseñados
específicamente para LPP (Lesiones Por Presión) y compliance HIPAA.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PromptTemplate(Enum):
    """Templates de prompts médicos disponibles"""
    LPP_STAGING = "lpp_staging"
    WOUND_ASSESSMENT = "wound_assessment" 
    RISK_STRATIFICATION = "risk_stratification"
    INTERVENTION_PLANNING = "intervention_planning"
    CLINICAL_DOCUMENTATION = "clinical_documentation"
    PAIN_ASSESSMENT = "pain_assessment"
    INFECTION_SCREENING = "infection_screening"


@dataclass
class MedicalPromptContext:
    """Contexto para personalización de prompts médicos"""
    patient_age_group: Optional[str] = None  # "pediatric", "adult", "geriatric"
    care_setting: Optional[str] = None  # "home", "hospital", "ltc", "ambulatory"
    language_preference: str = "es"  # "es", "en"
    clinical_urgency: Optional[str] = None  # "routine", "urgent", "emergent"
    provider_type: Optional[str] = None  # "nurse", "physician", "specialist"


class MedicalPromptBuilder:
    """Constructor de prompts médicos especializados para Vigia"""
    
    @staticmethod
    def build_lpp_staging_prompt(
        clinical_observations: str,
        image_findings: Optional[Dict[str, Any]] = None,
        context: Optional[MedicalPromptContext] = None
    ) -> str:
        """
        Construir prompt para estadificación de LPP.
        
        Args:
            clinical_observations: Observaciones clínicas del proveedor
            image_findings: Hallazgos del análisis de imagen (opcional)
            context: Contexto médico para personalización (opcional)
            
        Returns:
            Prompt estructurado para análisis de estadificación LPP
        """
        
        language = context.language_preference if context else "es"
        
        if language == "es":
            base_prompt = """
Como especialista en medicina interna y cuidado de heridas, analiza los siguientes hallazgos para determinar el estadio de la lesión por presión según la clasificación NPUAP/EPUAP actualizada.

HALLAZGOS CLÍNICOS:
{clinical_observations}

SISTEMA DE CLASIFICACIÓN NPUAP/EPUAP:
- Estadio 1: Eritema no blanqueable en piel intacta
- Estadio 2: Pérdida de espesor parcial con lecho de herida viable
- Estadio 3: Pérdida de espesor total del tejido
- Estadio 4: Pérdida de espesor total con exposición ósea/tendinosa
- Lesión Tisular Profunda Sospechada: Área localizada de decoloración púrpura-marronácea
- No Clasificable: Pérdida de espesor total con base oscurecida

CRITERIOS DE EVALUACIÓN:
1. Integridad de la piel y profundidad de la lesión
2. Características del lecho de la herida
3. Presencia y tipo de tejido no viable
4. Signos de infección o complicaciones
5. Localización anatómica y factores contribuyentes

FORMATO DE RESPUESTA REQUERIDO:
{{
    "estadio_lpp": "Estadio específico según NPUAP/EPUAP",
    "confianza_diagnostica": 0.85,
    "caracteristicas_clave": ["característica1", "característica2"],
    "diametro_aproximado": "medidas si disponibles",
    "profundidad": "descripción de profundidad",
    "lecho_herida": "descripción del lecho",
    "bordes_herida": "descripción de bordes",
    "exudado": "tipo y cantidad de exudado",
    "signos_infeccion": ["signo1", "signo2"] or [],
    "factores_contribuyentes": ["factor1", "factor2"],
    "recomendaciones_inmediatas": ["recomendación1", "recomendación2"],
    "seguimiento_recomendado": "frecuencia de evaluación",
    "notas_clinicas": "observaciones adicionales importantes"
}}
"""
        else:  # English
            base_prompt = """
As a specialist in internal medicine and wound care, analyze the following findings to determine the pressure injury stage according to the updated NPUAP/EPUAP classification.

CLINICAL FINDINGS:
{clinical_observations}

NPUAP/EPUAP CLASSIFICATION SYSTEM:
- Stage 1: Non-blanchable erythema of intact skin
- Stage 2: Partial-thickness skin loss with viable wound bed
- Stage 3: Full-thickness tissue loss
- Stage 4: Full-thickness tissue loss with exposed bone/tendon
- Suspected Deep Tissue Injury: Localized area of purple-maroon discoloration
- Unstageable: Full-thickness tissue loss with obscured base

EVALUATION CRITERIA:
1. Skin integrity and depth of injury
2. Wound bed characteristics
3. Presence and type of non-viable tissue
4. Signs of infection or complications
5. Anatomical location and contributing factors

REQUIRED RESPONSE FORMAT:
{{
    "pressure_injury_stage": "Specific stage per NPUAP/EPUAP",
    "diagnostic_confidence": 0.85,
    "key_characteristics": ["characteristic1", "characteristic2"],
    "approximate_diameter": "measurements if available",
    "depth": "depth description",
    "wound_bed": "wound bed description",
    "wound_edges": "edge description",
    "exudate": "type and amount of exudate",
    "infection_signs": ["sign1", "sign2"] or [],
    "contributing_factors": ["factor1", "factor2"],
    "immediate_recommendations": ["recommendation1", "recommendation2"],
    "recommended_followup": "evaluation frequency",
    "clinical_notes": "additional important observations"
}}
"""

        # Personalizar según contexto
        if context:
            if context.care_setting == "home":
                base_prompt += "\n\nCONSIDERACIONES ESPECIALES:\n- Adapta recomendaciones para cuidado domiciliario\n- Considera recursos disponibles en el hogar\n- Incluye educación para paciente/cuidador"
            elif context.care_setting == "hospital":
                base_prompt += "\n\nCONSIDERACIONES ESPECIALES:\n- Considera protocolos hospitalarios\n- Evalúa necesidad de consulta especializada\n- Incluye consideraciones para continuidad del cuidado"

        # Incluir hallazgos de imagen si están disponibles
        if image_findings:
            image_section = f"""

HALLAZGOS DE ANÁLISIS DE IMAGEN (Computer Vision):
- Detecciones automáticas: {image_findings.get('detections', 'No disponible')}
- Características visuales: {image_findings.get('visual_features', 'No disponible')}
- Mediciones automáticas: {image_findings.get('measurements', 'No disponible')}
- Confianza del algoritmo: {image_findings.get('confidence_score', 'No disponible')}

NOTA: Los hallazgos de imagen deben correlacionarse con la evaluación clínica directa.
"""
            base_prompt = base_prompt.replace("CRITERIOS DE EVALUACIÓN:", image_section + "\nCRITERIOS DE EVALUACIÓN:")

        return base_prompt.format(clinical_observations=clinical_observations)

    @staticmethod
    def build_risk_assessment_prompt(
        patient_data: Dict[str, Any],
        current_findings: Optional[str] = None,
        context: Optional[MedicalPromptContext] = None
    ) -> str:
        """
        Construir prompt para evaluación de riesgo de LPP.
        """
        
        language = context.language_preference if context else "es"
        
        if language == "es":
            prompt = """
Como especialista en prevención de lesiones por presión, evalúa el riesgo de desarrollo de nuevas LPP basándote en los siguientes datos del paciente.

DATOS DEL PACIENTE:
{patient_data}

ESCALAS DE EVALUACIÓN A CONSIDERAR:
1. Escala de Braden (Percepción sensorial, Humedad, Actividad, Movilidad, Nutrición, Fricción/Cizallamiento)
2. Escala de Norton (Condición física, Estado mental, Actividad, Movilidad, Incontinencia)
3. Factores de riesgo adicionales específicos

FACTORES DE RIESGO PRINCIPALES:
- Inmovilidad/actividad limitada
- Incontinencia urinaria/fecal
- Malnutrición/deshidratación
- Alteraciones de la conciencia
- Perfusión/oxigenación comprometida
- Edad avanzada
- Medicamentos (sedantes, vasopresores)
- Condiciones médicas (diabetes, enfermedad vascular)

FORMATO DE RESPUESTA:
{{
    "puntuacion_braden": "puntuación numérica si calculable",
    "nivel_riesgo": "bajo/moderado/alto/muy alto",
    "confianza_evaluacion": 0.88,
    "factores_riesgo_presentes": ["factor1", "factor2"],
    "factores_protectores": ["factor1", "factor2"],
    "recomendaciones_prevencion": [
        {{
            "categoria": "reposicionamiento",
            "frecuencia": "cada 2 horas",
            "especificaciones": "detalles específicos"
        }}
    ],
    "monitorizacion_recomendada": "frecuencia de evaluación",
    "educacion_necesaria": ["tema1", "tema2"],
    "equipo_requerido": ["equipo1", "equipo2"],
    "seguimiento": "plan de seguimiento"
}}
"""
        else:  # English
            prompt = """
As a pressure injury prevention specialist, assess the risk of developing new pressure injuries based on the following patient data.

PATIENT DATA:
{patient_data}

ASSESSMENT SCALES TO CONSIDER:
1. Braden Scale (Sensory perception, Moisture, Activity, Mobility, Nutrition, Friction/Shear)
2. Norton Scale (Physical condition, Mental state, Activity, Mobility, Incontinence)
3. Additional specific risk factors

MAIN RISK FACTORS:
- Immobility/limited activity
- Urinary/fecal incontinence
- Malnutrition/dehydration
- Altered consciousness
- Compromised perfusion/oxygenation
- Advanced age
- Medications (sedatives, vasopressors)
- Medical conditions (diabetes, vascular disease)

RESPONSE FORMAT:
{{
    "braden_score": "numerical score if calculable",
    "risk_level": "low/moderate/high/very high",
    "assessment_confidence": 0.88,
    "present_risk_factors": ["factor1", "factor2"],
    "protective_factors": ["factor1", "factor2"],
    "prevention_recommendations": [
        {{
            "category": "repositioning",
            "frequency": "every 2 hours",
            "specifications": "specific details"
        }}
    ],
    "recommended_monitoring": "evaluation frequency",
    "required_education": ["topic1", "topic2"],
    "required_equipment": ["equipment1", "equipment2"],
    "followup": "followup plan"
}}
"""

        if current_findings:
            findings_section = f"""

HALLAZGOS ACTUALES:
{current_findings}

NOTA: Considera estos hallazgos en el contexto de prevención de nuevas lesiones.
"""
            prompt = prompt.replace("FORMATO DE RESPUESTA:", findings_section + "\nFORMATO DE RESPUESTA:")

        return prompt.format(patient_data=str(patient_data))

    @staticmethod
    def build_intervention_prompt(
        lpp_stage: str,
        patient_factors: Dict[str, Any],
        care_setting: str = "home",
        context: Optional[MedicalPromptContext] = None
    ) -> str:
        """
        Construir prompt para planificación de intervenciones.
        """
        
        language = context.language_preference if context else "es"
        
        if language == "es":
            prompt = """
Como especialista en cuidado de heridas, desarrolla un plan de intervención integral para el manejo de lesión por presión.

DIAGNÓSTICO:
Lesión por Presión - {lpp_stage}

FACTORES DEL PACIENTE:
{patient_factors}

ENTORNO DE CUIDADO:
{care_setting}

PRINCIPIOS DE MANEJO DE LPP:
1. Alivio de presión (reposicionamiento, superficies especiales)
2. Manejo de la herida (limpieza, desbridamiento, apósitos)
3. Manejo de la carga bacteriana e infección
4. Apoyo nutricional
5. Manejo del dolor
6. Educación del paciente/cuidador

CONSIDERACIONES POR ESTADIO:
- Estadio 1: Protección, alivio presión, hidratación
- Estadio 2: Ambiente húmedo, protección de trauma
- Estadio 3-4: Desbridamiento, manejo exudado, promoción granulación
- No clasificable: Evaluación especializada, posible desbridamiento

FORMATO DE RESPUESTA:
{{
    "plan_manejo": {{
        "alivio_presion": {{
            "reposicionamiento": "frecuencia y técnica",
            "superficies_soporte": "tipo recomendado",
            "dispositivos_proteccion": ["dispositivo1", "dispositivo2"]
        }},
        "cuidado_herida": {{
            "limpieza": "solución y técnica",
            "desbridamiento": "si es necesario y tipo",
            "apositos": "tipo y frecuencia de cambio",
            "monitorizacion": "parámetros a observar"
        }},
        "manejo_sistemico": {{
            "nutricion": "recomendaciones específicas",
            "hidratacion": "objetivos de hidratación",
            "medicamentos": "analgesia u otros",
            "comorbilidades": "manejo de condiciones asociadas"
        }}
    }},
    "objetivos_tratamiento": ["objetivo1", "objetivo2"],
    "cronograma_evaluacion": "frecuencia de seguimiento",
    "criterios_derivacion": ["criterio1", "criterio2"],
    "educacion_paciente": ["tema1", "tema2"],
    "recursos_necesarios": ["recurso1", "recurso2"],
    "indicadores_progreso": ["indicador1", "indicador2"],
    "plan_contingencia": "qué hacer si no hay mejoría"
}}
"""
        else:  # English
            prompt = """
As a wound care specialist, develop a comprehensive intervention plan for pressure injury management.

DIAGNOSIS:
Pressure Injury - {lpp_stage}

PATIENT FACTORS:
{patient_factors}

CARE SETTING:
{care_setting}

PRESSURE INJURY MANAGEMENT PRINCIPLES:
1. Pressure relief (repositioning, special surfaces)
2. Wound management (cleaning, debridement, dressings)
3. Bacterial load and infection management
4. Nutritional support
5. Pain management
6. Patient/caregiver education

STAGE-SPECIFIC CONSIDERATIONS:
- Stage 1: Protection, pressure relief, moisturization
- Stage 2: Moist environment, trauma protection
- Stage 3-4: Debridement, exudate management, granulation promotion
- Unstageable: Specialist evaluation, possible debridement

RESPONSE FORMAT:
{{
    "management_plan": {{
        "pressure_relief": {{
            "repositioning": "frequency and technique",
            "support_surfaces": "recommended type",
            "protection_devices": ["device1", "device2"]
        }},
        "wound_care": {{
            "cleansing": "solution and technique",
            "debridement": "if needed and type",
            "dressings": "type and change frequency",
            "monitoring": "parameters to observe"
        }},
        "systemic_management": {{
            "nutrition": "specific recommendations",
            "hydration": "hydration goals",
            "medications": "analgesia or others",
            "comorbidities": "management of associated conditions"
        }}
    }},
    "treatment_goals": ["goal1", "goal2"],
    "evaluation_schedule": "followup frequency",
    "referral_criteria": ["criterion1", "criterion2"],
    "patient_education": ["topic1", "topic2"],
    "required_resources": ["resource1", "resource2"],
    "progress_indicators": ["indicator1", "indicator2"],
    "contingency_plan": "what to do if no improvement"
}}
"""

        return prompt.format(
            lpp_stage=lpp_stage,
            patient_factors=str(patient_factors),
            care_setting=care_setting
        )

    @staticmethod
    def build_pain_assessment_prompt(
        patient_report: str,
        behavioral_observations: Optional[str] = None,
        context: Optional[MedicalPromptContext] = None
    ) -> str:
        """
        Construir prompt para evaluación del dolor en LPP.
        """
        
        language = context.language_preference if context else "es"
        
        if language == "es":
            prompt = """
Como especialista en manejo del dolor en heridas, evalúa el dolor asociado con lesión por presión basándote en la información proporcionada.

REPORTE DEL PACIENTE:
{patient_report}

ESCALAS DE DOLOR RELEVANTES:
- Escala Visual Analógica (EVA): 0-10
- Escala de Caras (para pacientes con dificultades comunicativas)
- Escala PAINAD (para pacientes con demencia)
- Escala CNPI (Checklist of Nonverbal Pain Indicators)

TIPOS DE DOLOR EN LPP:
1. Dolor nociceptivo: daño tisular directo
2. Dolor neuropático: daño nervioso
3. Dolor por procedimientos: durante cuidados
4. Dolor por presión: relacionado con posicionamiento

FORMATO DE RESPUESTA:
{{
    "intensidad_dolor": "nivel 0-10 o descripción",
    "tipo_dolor": "nociceptivo/neuropático/mixto",
    "caracteristicas": ["punzante", "urente", "constante"],
    "factores_agravantes": ["movimiento", "toque", "cambio_posicion"],
    "factores_aliviantes": ["reposo", "posicion_especifica"],
    "impacto_funcional": "descripción del impacto en AVD",
    "recomendaciones_farmacologicas": [
        {{
            "medicamento": "nombre",
            "dosis": "dosis recomendada",
            "frecuencia": "frecuencia",
            "duracion": "duración del tratamiento"
        }}
    ],
    "medidas_no_farmacologicas": ["medida1", "medida2"],
    "modificaciones_cuidado": ["modificación1", "modificación2"],
    "seguimiento_dolor": "frecuencia de evaluación"
}}
"""
        else:  # English
            prompt = """
As a specialist in wound pain management, assess pain associated with pressure injury based on the provided information.

PATIENT REPORT:
{patient_report}

RELEVANT PAIN SCALES:
- Visual Analog Scale (VAS): 0-10
- Faces Scale (for patients with communication difficulties)
- PAINAD Scale (for patients with dementia)
- CNPI Scale (Checklist of Nonverbal Pain Indicators)

TYPES OF PAIN IN PRESSURE INJURIES:
1. Nociceptive pain: direct tissue damage
2. Neuropathic pain: nerve damage
3. Procedural pain: during care procedures
4. Pressure pain: positioning-related

RESPONSE FORMAT:
{{
    "pain_intensity": "level 0-10 or description",
    "pain_type": "nociceptive/neuropathic/mixed",
    "characteristics": ["stabbing", "burning", "constant"],
    "aggravating_factors": ["movement", "touch", "position_change"],
    "relieving_factors": ["rest", "specific_position"],
    "functional_impact": "impact on ADL description",
    "pharmacological_recommendations": [
        {{
            "medication": "name",
            "dose": "recommended dose",
            "frequency": "frequency",
            "duration": "treatment duration"
        }}
    ],
    "non_pharmacological_measures": ["measure1", "measure2"],
    "care_modifications": ["modification1", "modification2"],
    "pain_followup": "evaluation frequency"
}}
"""

        if behavioral_observations:
            behavioral_section = f"""

OBSERVACIONES CONDUCTUALES:
{behavioral_observations}

NOTA: Considera signos no verbales de dolor, especialmente en pacientes con limitaciones comunicativas.
"""
            prompt = prompt.replace("FORMATO DE RESPUESTA:", behavioral_section + "\nFORMATO DE RESPUESTA:")

        return prompt.format(patient_report=patient_report)

    @staticmethod 
    def get_specialized_prompt(
        template: PromptTemplate,
        content: str,
        additional_context: Optional[Dict[str, Any]] = None,
        prompt_context: Optional[MedicalPromptContext] = None
    ) -> str:
        """
        Obtener prompt especializado según el template solicitado.
        
        Args:
            template: Tipo de template médico
            content: Contenido principal para análisis
            additional_context: Contexto adicional específico (opcional)
            prompt_context: Contexto para personalización (opcional)
            
        Returns:
            Prompt especializado y personalizado
        """
        
        if template == PromptTemplate.LPP_STAGING:
            return MedicalPromptBuilder.build_lpp_staging_prompt(
                content, additional_context, prompt_context
            )
        elif template == PromptTemplate.RISK_STRATIFICATION:
            return MedicalPromptBuilder.build_risk_assessment_prompt(
                {"clinical_data": content}, content, prompt_context
            )
        elif template == PromptTemplate.INTERVENTION_PLANNING:
            lpp_stage = additional_context.get("lpp_stage", "Unknown") if additional_context else "Unknown"
            patient_factors = additional_context.get("patient_factors", {}) if additional_context else {}
            care_setting = additional_context.get("care_setting", "home") if additional_context else "home"
            
            return MedicalPromptBuilder.build_intervention_prompt(
                lpp_stage, patient_factors, care_setting, prompt_context
            )
        elif template == PromptTemplate.PAIN_ASSESSMENT:
            behavioral_obs = additional_context.get("behavioral_observations") if additional_context else None
            return MedicalPromptBuilder.build_pain_assessment_prompt(
                content, behavioral_obs, prompt_context
            )
        else:
            # Fallback to basic medical analysis prompt
            return f"""
Como especialista médico, analiza la siguiente información clínica:

{content}

Proporciona una evaluación médica estructurada con recomendaciones basadas en evidencia.
"""