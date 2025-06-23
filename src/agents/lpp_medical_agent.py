"""
Vigía Medical Agent - ADK Agent for Pressure Injury Detection and Management
Based on customer-service agent pattern from ADK samples.
"""

import logging
import warnings
from datetime import datetime

# Try to import ADK Agent, fall back to mock if not available
try:
    from google.adk.agents.llm_agent import Agent
except ImportError:
    # Mock Agent class for testing when google.adk is not available
    class Agent:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'MockAgent')
            self.instructions = kwargs.get('instructions', '')
            self.tools = kwargs.get('tools', [])
        
        def run(self, prompt):
            return f"Mock response for: {prompt}"

# Import our Slack tools
from vigia_detect.messaging.adk_tools import enviar_alerta_lpp, test_slack_desde_adk

warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")

# Configure logging
logger = logging.getLogger(__name__)

# Agent configuration
LPP_GLOBAL_INSTRUCTION = """
Eres un agente médico especializado en detección y prevención de Lesiones por Presión (LPP).
Tu función principal es:

1. Analizar resultados de detección de LPP del pipeline de visión computacional
2. Clasificar severidad según criterios EPUAP/NPIAP (Grados 0-4)
3. Notificar al equipo médico vía Slack según protocolos de urgencia
4. Mantener registros auditables para cumplimiento normativo
5. Proporcionar recomendaciones basadas en evidencia médica

PROTOCOLOS DE NOTIFICACIÓN:
- Grado 0 (Sin LPP): Notificación informativa
- Grado 1 (Eritema): Atención programada
- Grado 2 (Úlcera superficial): Atención pronta
- Grado 3 (Úlcera profunda): CRÍTICO - Evaluación inmediata
- Grado 4 (Úlcera hasta hueso): EMERGENCIA - Acción inmediata

Siempre mantén confidencialidad médica y usa identificadores anonimizados.
"""

LPP_INSTRUCTION = """
Como agente médico de Vigía, procesas detecciones de lesiones por presión y coordinas respuesta médica.

FLUJO DE TRABAJO:
1. Recibir resultados análisis CV pipeline
2. Interpretar confianza y características detectadas
3. Clasificar severidad LPP (0-4)
4. Determinar urgencia según protocolo
5. Enviar notificación Slack apropiada
6. Documentar acción para auditoría

CRITERIOS CLASIFICACIÓN LPP:
- Grado 0: Piel intacta, sin eritema
- Grado 1: Eritema no blanqueable, piel intacta
- Grado 2: Pérdida parcial espesor, úlcera superficial
- Grado 3: Pérdida total espesor, grasa visible
- Grado 4: Pérdida total espesor, hueso/músculo expuesto

Usa siempre terminología médica precisa y mantén tono profesional.
"""

def procesar_imagen_lpp(imagen_path: str, paciente_id: str, 
                       resultados_cv: dict) -> dict:
    """
    Procesa resultados de detección LPP del pipeline CV.
    
    Args:
        imagen_path: Path de la imagen analizada
        paciente_id: ID anonimizado del paciente  
        resultados_cv: Dict con resultados del modelo YOLOv5:
            - detection_class: Clase detectada (0-4)
            - confidence: Confianza del modelo (0-1)
            - bbox: Coordenadas bounding box
            - timestamp: Timestamp análisis
    
    Returns:
        dict: Resultado procesamiento para el agente
    """
    logger.info(f"Procesando imagen LPP - Paciente: {paciente_id}")
    
    severidad = resultados_cv.get('detection_class', 0)
    confidence = resultados_cv.get('confidence', 0.0)
    timestamp = resultados_cv.get('timestamp', datetime.now().isoformat())
    
    # Determinar ubicación anatómica (mock - en producción vendría del contexto)
    ubicacion = "Zona analizada"  # Placeholder
    
    # Construir detalles para notificación
    detalles = {
        'confidence': round(confidence * 100, 1),
        'timestamp': timestamp,
        'ubicacion': ubicacion,
        'imagen_path': imagen_path,
        'bbox': resultados_cv.get('bbox', [])
    }
    
    logger.info(f"LPP procesada - Severidad: {severidad}, Confianza: {confidence:.1%}")
    
    return {
        'status': 'processed',
        'severidad': severidad,
        'paciente_id': paciente_id,
        'detalles': detalles,
        'require_notification': severidad > 0  # Solo notificar si hay LPP detectada
    }

def generar_reporte_lpp(paciente_id: str, severidad: int, 
                       confianza: float) -> dict:
    """
    Genera reporte médico estructurado de detección LPP.
    
    Args:
        paciente_id: ID paciente anonimizado
        severidad: Grado LPP (0-4)
        confianza: Confianza detección (0-1)
    
    Returns:
        dict: Reporte médico estructurado
    """
    
    clasificaciones = {
        0: "Sin evidencia LPP",
        1: "LPP Grado I - Eritema no blanqueable", 
        2: "LPP Grado II - Úlcera espesor parcial",
        3: "LPP Grado III - Úlcera espesor total",
        4: "LPP Grado IV - Úlcera espesor total con exposición"
    }
    
    urgencias = {
        0: "Rutina",
        1: "Seguimiento programado", 
        2: "Atención pronta",
        3: "Crítico - Evaluación inmediata",
        4: "Emergencia - Acción inmediata"
    }
    
    reporte = {
        'paciente_id': paciente_id,
        'timestamp': datetime.now().isoformat(),
        'clasificacion': clasificaciones.get(severidad, "Clasificación desconocida"),
        'severidad': severidad,
        'confianza_deteccion': f"{confianza:.1%}",
        'nivel_urgencia': urgencias.get(severidad, "Sin definir"),
        'requiere_intervencion': severidad >= 2,
        'seguimiento_requerido': severidad >= 1
    }
    
    logger.info(f"Reporte LPP generado - {reporte['clasificacion']}")
    return reporte

# Create LPP Medical Agent
lpp_agent = Agent(
    model="gemini-2.0-flash-exp",  # Latest model for medical analysis
    global_instruction=LPP_GLOBAL_INSTRUCTION,
    instruction=LPP_INSTRUCTION,
    name="lpp_medical_agent",
    tools=[
        procesar_imagen_lpp,
        generar_reporte_lpp,
        enviar_alerta_lpp,  # Slack notification tool
        test_slack_desde_adk,  # Slack testing tool
    ],
    # Add callbacks for medical logging if needed
    # before_tool_callback=medical_audit_callback,
    # after_tool_callback=medical_logging_callback,
)

# Export agent for use
__all__ = ['lpp_agent', 'procesar_imagen_lpp', 'generar_reporte_lpp']
