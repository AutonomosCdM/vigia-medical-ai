"""
Templates centralizados para mensajes y modales de Slack
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from .constants import SlackActionIds, LPP_SEVERITY_ALERTS, TEST_PATIENT_DATA


def create_detection_blocks(detection_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create Slack blocks for detection notification."""
    patient_code = detection_data.get('patient_code', 'Unknown')
    lpp_grade = detection_data.get('lpp_grade', 0)
    confidence = detection_data.get('confidence', 0.0)
    
    severity_info = LPP_SEVERITY_ALERTS.get(lpp_grade, LPP_SEVERITY_ALERTS[0])
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{severity_info['emoji']} Detección LPP - {patient_code}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Grado:* {lpp_grade}"
                },
                {
                    "type": "mrkdwn", 
                    "text": f"*Confianza:* {confidence:.2f}"
                }
            ]
        }
    ]
    
    return blocks


def create_error_blocks(error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create Slack blocks for error notification."""
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "❌ Error en Procesamiento"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Error: {error_data.get('message', 'Unknown error')}"
            }
        }
    ]
    
    return blocks


def create_patient_history_blocks(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create Slack blocks for patient history."""
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📋 Historial - {patient_data.get('patient_code', 'Unknown')}"
            }
        }
    ]
    
    return blocks


def create_detection_notification(detection_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create detection notification blocks - alias for create_detection_blocks."""
    return create_detection_blocks(detection_data)


class SlackModalTemplates:
    """Templates para modales de Slack"""
    
    @staticmethod
    def historial_medico(patient_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Crea el modal de historial médico.
        
        Args:
            patient_data: Datos del paciente. Si es None, usa datos de prueba.
        
        Returns:
            Estructura del modal para Slack
        """
        data = patient_data or TEST_PATIENT_DATA
        
        # Formatear diagnósticos
        diagnosticos = "\n".join([f"• {d}" for d in data.get("diagnoses", [])])
        
        # Formatear medicamentos
        medicamentos = "\n".join([f"• {m}" for m in data.get("medications", [])])
        
        # Formatear historial de LPP
        historial_lpp = []
        for h in data.get("lpp_history", []):
            historial_lpp.append(
                f"• *{h['date']}*: Grado {h['grade']} en {h['location']} - {h['status']}"
            )
        historial_text = "\n".join(historial_lpp) if historial_lpp else "Sin historial previo"
        
        return {
            "type": "modal",
            "callback_id": SlackActionIds.MODAL_HISTORIAL,
            "title": {
                "type": "plain_text",
                "text": "Historial Médico"
            },
            "close": {
                "type": "plain_text",
                "text": "Cerrar"
            },
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📋 {data['name']}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*ID Paciente:*\n{data['id']}"},
                        {"type": "mrkdwn", "text": f"*Edad:*\n{data['age']} años"},
                        {"type": "mrkdwn", "text": f"*Servicio:*\n{data['service']}"},
                        {"type": "mrkdwn", "text": f"*Cama:*\n{data['bed']}"}
                    ]
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🏥 Diagnósticos Actuales:*\n{diagnosticos}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*💊 Medicación Actual:*\n{medicamentos}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📊 Úlceras por Presión - Historial:*\n{historial_text}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🔄 Última Movilización:*\n{datetime.now().strftime('%Y-%m-%d %H:%M')} hrs"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📝 Observaciones:*\nPaciente con movilidad reducida post-quirúrgica. " +
                               "Requiere cambios posturales c/2h. Piel frágil, alto riesgo de UPP."
                    }
                }
            ]
        }
    
    @staticmethod
    def resolucion_caso() -> Dict[str, Any]:
        """Modal para marcar un caso como resuelto"""
        return {
            "type": "modal",
            "callback_id": SlackActionIds.MODAL_RESOLUCION,
            "title": {
                "type": "plain_text",
                "text": "Marcar como Resuelto"
            },
            "submit": {
                "type": "plain_text",
                "text": "Confirmar"
            },
            "close": {
                "type": "plain_text",
                "text": "Cancelar"
            },
            "blocks": [
                {
                    "type": "input",
                    "block_id": "resolucion_input",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "resolucion_text",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Describa las acciones tomadas y el resultado..."
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Resolución del Caso"
                    }
                },
                {
                    "type": "input",
                    "block_id": "tiempo_resolucion",
                    "element": {
                        "type": "static_select",
                        "action_id": "tiempo_select",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Seleccione tiempo de resolución"
                        },
                        "options": [
                            {
                                "text": {"type": "plain_text", "text": "< 30 minutos"},
                                "value": "30min"
                            },
                            {
                                "text": {"type": "plain_text", "text": "30 min - 1 hora"},
                                "value": "1hr"
                            },
                            {
                                "text": {"type": "plain_text", "text": "1 - 2 horas"},
                                "value": "2hr"
                            },
                            {
                                "text": {"type": "plain_text", "text": "> 2 horas"},
                                "value": "more"
                            }
                        ]
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Tiempo de Resolución"
                    }
                }
            ]
        }


class SlackMessageTemplates:
    """Templates para mensajes de Slack"""
    
    @staticmethod
    def botones_acciones_enfermeria() -> List[Dict[str, Any]]:
        """Botones estándar para acciones de enfermería"""
        return [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "📋 Ver Historial Médico"},
                "action_id": SlackActionIds.VER_HISTORIAL,
                "style": "primary"
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "🏥 Solicitar Evaluación Médica"},
                "action_id": SlackActionIds.SOLICITAR_EVALUACION,
                "style": "danger"
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "✅ Marcar como Resuelto"},
                "action_id": SlackActionIds.MARCAR_RESUELTO
            }
        ]
    
    @staticmethod
    def alerta_lpp(
        grado: int,
        paciente: str,
        id_caso: str,
        ubicacion: str,
        confianza: float,
        servicio: str,
        cama: str,
        imagen_url: Optional[str] = None,
        analisis_emocional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crea mensaje de alerta de LPP.
        
        Args:
            grado: Grado de la lesión (0-4)
            paciente: Nombre del paciente
            id_caso: ID del caso
            ubicacion: Ubicación anatómica
            confianza: Confianza del modelo (0-1)
            servicio: Servicio médico
            cama: Número de cama
            imagen_url: URL de la imagen (opcional)
            analisis_emocional: Análisis emocional del paciente (opcional)
            
        Returns:
            Estructura del mensaje para Slack
        """
        severity = LPP_SEVERITY_ALERTS.get(grado, LPP_SEVERITY_ALERTS[0])
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity['emoji']} NUEVO CASO VIGÍA - {severity['level']}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Paciente:*\n{paciente}"},
                    {"type": "mrkdwn", "text": f"*ID Caso:*\n{id_caso}"},
                    {"type": "mrkdwn", "text": f"*Servicio:*\n{servicio}"},
                    {"type": "mrkdwn", "text": f"*Cama:*\n{cama}"},
                    {"type": "mrkdwn", "text": f"*Fecha/Hora:*\n{datetime.now().strftime('%d/%m/%Y %H:%M')}"},
                    {"type": "mrkdwn", "text": f"*Estado:*\n🔴 Pendiente"}
                ]
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🔍 Detección de Lesión por Presión*\n\n" +
                           f"• *Grado:* {severity['emoji']} Grado {grado}\n" +
                           f"• *Ubicación anatómica:* {ubicacion}\n" +
                           f"• *Confianza del modelo:* {confianza:.1%}\n" +
                           f"• *Descripción:* {severity['message']}"
                }
            }
        ]
        
        # Agregar imagen si está disponible
        if imagen_url:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*📸 Imagen de la lesión:*"},
                "accessory": {
                    "type": "image",
                    "image_url": imagen_url,
                    "alt_text": "Imagen de lesión detectada"
                }
            })
        
        # Agregar análisis emocional si está disponible
        if analisis_emocional:
            preocupaciones = "\n".join([f"  - {p}" for p in analisis_emocional.get("preocupaciones", [])])
            blocks.extend([
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*💭 Análisis Emocional del Paciente*\n\n" +
                               f"• *Sentimiento detectado:* {analisis_emocional.get('sentimiento', 'No detectado')}\n" +
                               f"• *Estado de ánimo:* {analisis_emocional.get('estado_animo', 'No evaluado')}\n" +
                               f"• *Preocupaciones expresadas:*\n{preocupaciones}"
                    }
                }
            ])
        
        # Agregar botones de acción
        blocks.append({
            "type": "actions",
            "elements": SlackMessageTemplates.botones_acciones_enfermeria()
        })
        
        # Agregar contexto
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "Sistema Vigía v1.0 | Detección automática con IA"}
            ]
        })
        
        return {
            "blocks": blocks,
            "attachments": [{
                "color": severity['color']
            }]
        }