#!/usr/bin/env python3
"""
VIGIA Medical AI - Block Kit Medical Notifications #it_vigia
Notificaciones médicas usando Block Kit components avanzados
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

from src.messaging.slack_block_templates import (
    VigiaMessageTemplates, MedicalBlockContext, MedicalSeverity
)
from src.core.constants import LPPGrade

async def send_block_kit_medical_notifications():
    """Enviar notificaciones médicas usando Block Kit components"""
    print("🏗️ VIGIA MEDICAL AI - BLOCK KIT NOTIFICATIONS #it_vigia")
    print("=" * 60)
    
    if not SLACK_AVAILABLE:
        print("❌ Slack SDK no disponible")
        return False
    
    # Get token
    bot_token = os.getenv('SLACK_BOT_TOKEN')
    if not bot_token:
        print("❌ SLACK_BOT_TOKEN no configurado")
        return False
    
    # Initialize Slack client and templates
    client = WebClient(token=bot_token)
    templates = VigiaMessageTemplates()
    channel = "#it_vigia"
    
    try:
        # Test connection
        auth_test = client.auth_test()
        print(f"✅ Bot conectado: {auth_test['user']}")
        
        # 1. Sistema de activación con Block Kit
        print("\n🏗️ 1. Block Kit: Activación del sistema...")
        
        context1 = MedicalBlockContext(
            batman_token="BATMAN_ACTIVATION_001",
            session_id="activation_session",
            timestamp=datetime.now(),
            severity=MedicalSeverity.LOW,
            urgency_level="low",
            medical_context={
                "system_status": "fully_operational",
                "agents_active": 9,
                "channel": "#it_vigia"
            }
        )
        
        # Create system status blocks
        activation_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🩺 VIGIA Medical AI - Sistema Activado"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Sistema médico completamente operacional*\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n📱 Canal: #it_vigia configurado"
                },
                "accessory": {
                    "type": "image",
                    "image_url": "https://via.placeholder.com/75x75/36a64f/ffffff?text=✓",
                    "alt_text": "Sistema Activo"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "*🤖 Agentes Activos:*\n9 Agentes médicos"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*🔒 Seguridad:*\nPHI Tokenization activo"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*🏥 Compliance:*\nHIPAA + NPUAP/EPUAP"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*🎯 Estado:*\nListo para producción"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🔒 Batman Token: `{context1.batman_token}` | Sistema VIGIA Medical AI"
                    }
                ]
            }
        ]
        
        result1 = client.chat_postMessage(
            channel=channel,
            blocks=activation_blocks,
            text="VIGIA Medical AI Sistema Activado"  # Fallback text
        )
        
        if result1['ok']:
            print("✅ Block Kit activación enviado")
        else:
            print(f"❌ Error: {result1}")
            
        await asyncio.sleep(3)
        
        # 2. Detección LPP con Block Kit avanzado
        print("🏗️ 2. Block Kit: Detección LPP...")
        
        context2 = MedicalBlockContext(
            batman_token=f"BATMAN_LPP_{datetime.now().strftime('%H%M%S')}",
            session_id="lpp_detection_session",
            timestamp=datetime.now(),
            severity=MedicalSeverity.HIGH,
            urgency_level="high",
            medical_context={
                "lpp_grade": "Grade_2",
                "confidence": 0.873,
                "location": "region_sacra"
            }
        )
        
        lpp_blocks = templates.create_lpp_detection_alert(
            context=context2,
            lpp_grade=LPPGrade.GRADE_2,
            confidence=0.873,
            clinical_recommendation="Implementar protocolo prevención LPP inmediatamente. Redistribución de presión cada 2 horas.",
            evidence_level="A"
        )
        
        result2 = client.chat_postMessage(
            channel=channel,
            blocks=lpp_blocks,
            text="🚨 Detección LPP - Alerta Médica"
        )
        
        if result2['ok']:
            print("✅ Block Kit LPP enviado")
        else:
            print(f"❌ Error LPP: {result2}")
            
        await asyncio.sleep(3)
        
        # 3. Análisis de voz con Block Kit
        print("🏗️ 3. Block Kit: Análisis de voz...")
        
        context3 = MedicalBlockContext(
            batman_token=f"BATMAN_VOICE_{datetime.now().strftime('%H%M%S')}",
            session_id="voice_analysis_session",
            timestamp=datetime.now(),
            severity=MedicalSeverity.MODERATE,
            urgency_level="medium",
            medical_context={
                "pain_level": "moderate_high",
                "confidence": 0.918,
                "emotional_state": "distressed"
            }
        )
        
        voice_results = {
            "pain_level": "moderado-alto",
            "stress_indicators": ["ansiedad", "tensión_vocal"],
            "emotional_state": "preocupado",
            "confidence": 0.918,
            "recommendations": [
                "Evaluación inmediata del dolor",
                "Considerar ajuste medicación",
                "Apoyo psicológico recomendado"
            ]
        }
        
        voice_blocks = templates.create_voice_analysis_summary(
            context=context3,
            voice_results=voice_results
        )
        
        result3 = client.chat_postMessage(
            channel=channel,
            blocks=voice_blocks,
            text="🎙️ Análisis de Voz Médica Completado"
        )
        
        if result3['ok']:
            print("✅ Block Kit análisis de voz enviado")
        else:
            print(f"❌ Error voz: {result3}")
            
        await asyncio.sleep(3)
        
        # 4. Coordinación de equipo con Block Kit
        print("🏗️ 4. Block Kit: Coordinación de equipo...")
        
        context4 = MedicalBlockContext(
            batman_token=f"BATMAN_TEAM_{datetime.now().strftime('%H%M%S')}",
            session_id="team_coordination_session",
            timestamp=datetime.now(),
            severity=MedicalSeverity.CRITICAL,
            urgency_level="critical",
            medical_context={
                "case_priority": "CRITICAL",
                "lpp_grade": "Grade_3",
                "team_response_required": True
            }
        )
        
        coordination_data = {
            "alert_type": "team_coordination",
            "case_priority": "CRÍTICO",
            "required_specialists": ["Dermatología", "Cirugía Plástica", "Enfermería"],
            "estimated_response_time": "15 minutos",
            "escalation_level": "Nivel 2"
        }
        
        team_blocks = templates.create_team_coordination_alert(
            context=context4,
            coordination_data=coordination_data
        )
        
        result4 = client.chat_postMessage(
            channel=channel,
            blocks=team_blocks,
            text="👨‍⚕️ Coordinación Equipo Médico Activada"
        )
        
        if result4['ok']:
            print("✅ Block Kit coordinación enviado")
        else:
            print(f"❌ Error coordinación: {result4}")
            
        await asyncio.sleep(3)
        
        # 5. Protocolo de emergencia con Block Kit
        print("🏗️ 5. Block Kit: Protocolo de emergencia...")
        
        context5 = MedicalBlockContext(
            batman_token=f"BATMAN_EMERGENCY_{datetime.now().strftime('%H%M%S')}",
            session_id="emergency_protocol_session",
            timestamp=datetime.now(),
            severity=MedicalSeverity.CRITICAL,
            urgency_level="emergency",
            medical_context={
                "emergency_type": "LPP_Grade_4",
                "immediate_action_required": True,
                "surgical_evaluation": True
            }
        )
        
        emergency_data = {
            "emergency_type": "LPP Grado 4",
            "location": "Región sacra",
            "immediate_actions": [
                "Aislamiento de presión inmediato",
                "Evaluación quirúrgica urgente",
                "Protocolo de infección"
            ],
            "team_response_required": True,
            "estimated_surgery_time": "2 horas"
        }
        
        emergency_blocks = templates.create_emergency_protocol_alert(
            context=context5,
            emergency_data=emergency_data
        )
        
        result5 = client.chat_postMessage(
            channel=channel,
            blocks=emergency_blocks,
            text="🚨 Protocolo de Emergencia Médica Activado"
        )
        
        if result5['ok']:
            print("✅ Block Kit emergencia enviado")
        else:
            print(f"❌ Error emergencia: {result5}")
        
        # Summary
        results = [result1['ok'], result2['ok'], result3['ok'], result4['ok'], result5['ok']]
        successful = sum(results)
        
        print(f"\n📊 RESUMEN BLOCK KIT: {successful}/5 notificaciones enviadas")
        
        if successful == 5:
            print("🎉 ¡TODAS LAS NOTIFICACIONES BLOCK KIT ENVIADAS!")
            print("🏗️ Verificar #it_vigia para ver los componentes visuales")
            return True
        else:
            print(f"⚠️ {5-successful} notificaciones fallaron")
            return False
            
    except SlackApiError as e:
        print(f"❌ Error Slack API: {e.response['error']}")
        return False
    except Exception as e:
        print(f"❌ Error general: {e}")
        return False

async def main():
    """Ejecutar notificaciones Block Kit"""
    print("🚀 INICIANDO NOTIFICACIONES BLOCK KIT MÉDICAS")
    print("🏗️ VIGIA Medical AI → #it_vigia (Visual Components)")
    print("=" * 60)
    
    success = await send_block_kit_medical_notifications()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ¡BLOCK KIT MÉDICO COMPLETAMENTE DESPLEGADO!")
        print("🏗️ Notificaciones visuales avanzadas en #it_vigia")
        print("🏥 Componentes médicos interactivos funcionando")
    else:
        print("❌ Algunos componentes Block Kit fallaron")
    
    print("\n🏗️ COMPONENTES BLOCK KIT DESPLEGADOS:")
    print("   📊 Headers médicos con iconografía")
    print("   🎨 Secciones con campos estructurados")  
    print("   🖼️ Elementos visuales y accesorios")
    print("   🔘 Botones de acción médica")
    print("   📋 Contexto con Batman tokens")
    print("   🎯 Alertas con colores de severidad")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())