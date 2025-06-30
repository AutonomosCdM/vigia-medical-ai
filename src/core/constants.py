"""
Constantes y configuraciones compartidas del sistema Vigía
"""
from enum import Enum
from typing import Dict, Any, Final, ClassVar


class ProcessingRoute(Enum):
    """Medical processing routes for triage engine"""
    EMERGENCY = "emergency"          # Immediate medical attention required
    URGENT = "urgent"               # Urgent medical review needed
    ROUTINE = "routine"             # Standard medical workflow
    PREVENTIVE = "preventive"       # Preventive care recommendations
    REVIEW = "review"              # Requires medical professional review
    AUTOMATIC = "automatic"        # Can be processed automatically


class LPPGrade(Enum):
    """Grados de lesiones por presión según clasificación internacional"""
    NO_LESION = 0  # Sin lesión (eritema)
    GRADE_1 = 1    # Eritema no blanqueable
    GRADE_2 = 2    # Pérdida parcial del espesor de la piel
    GRADE_3 = 3    # Pérdida total del espesor de la piel
    GRADE_4 = 4    # Pérdida total del espesor de los tejidos


class SlackActionIds:
    """Action IDs estandarizados para interacciones de Slack"""
    # Botones principales
    VER_HISTORIAL: ClassVar[str] = "ver_historial_medico"
    SOLICITAR_EVALUACION: ClassVar[str] = "solicitar_evaluacion_medica"
    MARCAR_RESUELTO: ClassVar[str] = "marcar_resuelto"
    
    # Modales
    MODAL_HISTORIAL: ClassVar[str] = "modal_historial_medico"
    MODAL_RESOLUCION: ClassVar[str] = "modal_resolucion_caso"
    
    # Acciones secundarias
    ACEPTAR_EVALUACION: ClassVar[str] = "aceptar_evaluacion"
    VER_DETALLES_EVALUACION: ClassVar[str] = "ver_detalles_evaluacion"


class SlackChannels:
    """IDs de canales de Slack"""
    # Working channel (migrated from existing setup)
    DEFAULT: ClassVar[str] = "C08U2TB78E6"      # Working channel from environment
    
    # Legacy channels (preserved for compatibility)
    PROJECT_LPP: ClassVar[str] = "C08KK1SRE5S"  # #project-lpp
    VIGIA: ClassVar[str] = "C08TJHZFVD1"        # #vigia
    
    # Medical specialization channels (all route to default for now)
    EMERGENCY_ROOM: ClassVar[str] = "C08U2TB78E6"  # Use working channel
    CLINICAL_TEAM: ClassVar[str] = "C08U2TB78E6"   # Use working channel
    LPP_SPECIALISTS: ClassVar[str] = "C08U2TB78E6" # Use working channel
    NURSING_STAFF: ClassVar[str] = "C08U2TB78E6"   # Use working channel
    SYSTEM_ALERTS: ClassVar[str] = "C08U2TB78E6"   # Use working channel
    AUDIT_LOG: ClassVar[str] = "C08U2TB78E6"       # Use working channel
    
    # Método para obtener de configuración si existe
    @classmethod
    def get_channel(cls, name: str) -> str:
        """Obtiene el ID del canal, con fallback a valores por defecto"""
        import os
        # Try environment variable first, then class attribute, then default
        env_channel = os.getenv('SLACK_CHANNEL_IDS', cls.DEFAULT)
        return getattr(cls, name.upper(), env_channel)


# Mapeo de severidad de LPP para alertas
LPP_SEVERITY_ALERTS: Final[Dict[int, Dict[str, Any]]] = {
    0: {
        'emoji': '⚪',
        'level': 'INFO',
        'message': 'Sin LPP detectada - Eritema presente',
        'urgency': 'low',
        'color': '#36a64f'  # Verde
    },
    1: {
        'emoji': '🟡',
        'level': 'ATENCIÓN',
        'message': 'LPP Grado 1 - Eritema no blanqueable',
        'urgency': 'medium',
        'color': '#ff9f1a'  # Amarillo
    },
    2: {
        'emoji': '🟠',
        'level': 'IMPORTANTE',
        'message': 'LPP Grado 2 - Úlcera superficial',
        'urgency': 'high',
        'color': '#ff6b35'  # Naranja
    },
    3: {
        'emoji': '🔴',
        'level': 'URGENTE',
        'message': 'LPP Grado 3 - Úlcera profunda',
        'urgency': 'critical',
        'color': '#cc2936'  # Rojo
    },
    4: {
        'emoji': '⚫',
        'level': 'CRÍTICO',
        'message': 'LPP Grado 4 - Daño tisular extenso',
        'urgency': 'critical',
        'color': '#4a0e0e'  # Rojo oscuro
    }
}


# Descripciones detalladas de cada grado
LPP_GRADE_DESCRIPTIONS: Final[Dict[int, str]] = {
    0: "Categoría 1 (Eritema no blanqueable): Piel intacta con enrojecimiento no blanqueable.",
    1: "Categoría 1 (Eritema no blanqueable): Piel intacta con enrojecimiento no blanqueable de un área localizada.",
    2: "Categoría 2: Pérdida parcial del grosor de la piel que afecta a la epidermis y/o dermis.",
    3: "Categoría 3: Pérdida total del grosor de la piel que implica daño o necrosis del tejido subcutáneo.",
    4: "Categoría 4: Pérdida total del grosor de la piel con destrucción extensa, necrosis del tejido o daño muscular."
}


# Recomendaciones por grado
LPP_GRADE_RECOMMENDATIONS: Final[Dict[int, str]] = {
    0: "• Aliviar presión en zona afectada\n• Mantener piel limpia y seca\n• Cambios posturales c/2h",
    1: "• Aliviar presión inmediatamente\n• Aplicar apósitos protectores\n• Evaluar riesgo con escala Braden",
    2: "• Curación según protocolo\n• Apósitos hidrocoloides\n• Evaluar dolor y signos de infección",
    3: "• Evaluación médica urgente\n• Desbridamiento si necesario\n• Antibióticos según cultivo",
    4: "• Interconsulta cirugía plástica\n• Manejo multidisciplinario\n• Evaluación nutricional intensiva"
}


# Datos de prueba para desarrollo
TEST_PATIENT_DATA = {
    "name": "César Durán",
    "age": 45,
    "id": "CD-2025-001",
    "service": "Traumatología",
    "bed": "302-A",
    "diagnoses": [
        "Fractura de cadera derecha (post-operatorio)",
        "Diabetes Mellitus tipo 2",
        "Hipertensión arterial controlada"
    ],
    "medications": [
        "Metformina 850mg c/12h",
        "Losartán 50mg c/24h",
        "Tramadol 50mg c/8h PRN",
        "Omeprazol 20mg c/24h"
    ],
    "lpp_history": [
        {"date": "2025-01-15", "grade": 1, "location": "sacro", "status": "Resuelto"},
        {"date": "2025-02-01", "grade": 2, "location": "talón izq", "status": "En tratamiento"},
        {"date": "2025-02-22", "grade": 1, "location": "sacro", "status": "Nueva detección"}
    ]
}