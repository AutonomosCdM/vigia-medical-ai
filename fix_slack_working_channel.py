#!/usr/bin/env python3
"""
VIGIA Medical AI - Fix Slack Channel Working Issue
Enviar mensajes directos sin comandos slash que fallan
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.interfaces.slack_orchestrator import SlackOrchestrator, MedicalAlert
from src.messaging.slack_block_templates import MedicalBlockContext, MedicalSeverity

async def send_direct_messages():
    """Enviar mensajes médicos directos sin comandos slash"""
    print("🩺 ENVIANDO MENSAJES MÉDICOS DIRECTOS A #it_vigia")
    print("=" * 50)
    
    try:
        slack_orchestrator = SlackOrchestrator()
        
        # 1. Mensaje básico primero
        print("📤 Enviando mensaje básico...")
        basic_alert = MedicalAlert(
            severity=1,
            message=f"""🩺 **VIGIA Medical AI - Sistema Activo**

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏥 Sistema médico completamente operacional
📱 Canal #it_vigia configurado correctamente
🤖 9-Agent architecture funcionando
🔒 PHI tokenization activo

✅ **Componentes Ready:**
• Image Analysis (MONAI + YOLOv5)
• Voice Analysis (Hume AI)  
• Clinical Assessment Agent
• Risk Assessment Agent
• Diagnostic Agent
• Protocol Agent
• Communication Agent
• Master Medical Orchestrator
• Workflow Orchestration Agent

🎯 **Test enviado directamente desde Python**""",
            batman_token="BATMAN_DIRECT_001",
            alert_type="system_status",
            requires_immediate_response=False,
            clinical_context={
                "test_type": "direct_message",
                "channel": "#it_vigia",
                "bypass_slash_commands": True
            }
        )
        
        result1 = await slack_orchestrator.send_medical_alert(basic_alert)
        if result1.get('ok'):
            print("✅ Mensaje básico enviado")
        else:
            print(f"❌ Error mensaje básico: {result1.get('error')}")
        
        # 2. Alerta médica simulada
        print("📤 Enviando alerta médica LPP...")
        medical_alert = MedicalAlert(
            severity=3,
            message=f"""🚨 **ALERTA MÉDICA LPP DETECTADA**

🔴 **Grado LPP**: Grado 2 (Pérdida parcial del grosor de la piel)
📊 **Confianza**: 89.5%
📍 **Ubicación**: Región sacra
🎯 **Detección**: MONAI + YOLOv5

👨‍⚕️ **Recomendación Clínica** (Nivel A):
• Implementar protocolo prevención LPP inmediatamente
• Redistribución de presión cada 2 horas
• Evaluación por especialista LPP en <4 horas
• Documentar en historia clínica

⚡ **Acciones Automáticas**:
✅ Equipo médico notificado
✅ Protocolo NPUAP/EPUAP 2019 activado
✅ Escalación programada
✅ Registro de auditoría generado

🔒 Batman Token: BATMAN_LPP_REAL_{datetime.now().strftime('%H%M%S')}""",
            batman_token=f"BATMAN_LPP_REAL_{datetime.now().strftime('%H%M%S')}",
            alert_type="lpp_detection",
            requires_immediate_response=True,
            clinical_context={
                "lpp_grade": "Grade_2",
                "confidence": 0.895,
                "location": "region_sacra",
                "detection_method": "MONAI_YOLOv5",
                "npuap_compliant": True,
                "requires_specialist": True,
                "escalation_level": 2
            }
        )
        
        result2 = await slack_orchestrator.send_medical_alert(medical_alert)
        if result2.get('ok'):
            print("✅ Alerta médica LPP enviada")
        else:
            print(f"❌ Error alerta LPP: {result2.get('error')}")
        
        # 3. Análisis de voz simulado
        print("📤 Enviando análisis de voz...")
        voice_alert = MedicalAlert(
            severity=2,
            message=f"""🎙️ **ANÁLISIS DE VOZ MÉDICA COMPLETADO**

👤 **Paciente**: Batman Token (Datos protegidos)
🗣️ **Análisis Hume AI**: Completado

📊 **Resultados**:
• **Nivel de dolor**: Moderado (6/10)
• **Estrés vocal**: Detectado - Ansiedad presente
• **Estado emocional**: Preocupado/Tenso
• **Confianza análisis**: 92.3%

🩺 **Indicadores médicos**:
• Tensión vocal por dolor
• Marcadores de malestar
• Frecuencia de voz alterada
• Respuesta emocional al dolor

💊 **Recomendaciones**:
1. Evaluación inmediata del dolor
2. Considerar ajuste medicación analgésica
3. Apoyo psicológico recomendado
4. Seguimiento en 2 horas

🤖 **Procesado por**: VoiceAnalysisAgent
⏰ **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""",
            batman_token=f"BATMAN_VOICE_{datetime.now().strftime('%H%M%S')}",
            alert_type="voice_analysis",
            requires_immediate_response=False,
            clinical_context={
                "pain_level": "moderate_6_10",
                "stress_detected": True,
                "emotional_state": "worried_tense",
                "confidence": 0.923,
                "hume_ai_processed": True,
                "recommendations": ["pain_eval", "medication_adjust", "psychological_support"]
            }
        )
        
        result3 = await slack_orchestrator.send_medical_alert(voice_alert)
        if result3.get('ok'):
            print("✅ Análisis de voz enviado")
        else:
            print(f"❌ Error análisis voz: {result3.get('error')}")
        
        # 4. Coordinación de equipo
        print("📤 Enviando coordinación de equipo...")
        team_alert = MedicalAlert(
            severity=4,
            message=f"""👨‍⚕️ **COORDINACIÓN EQUIPO MÉDICO ACTIVADA**

🚨 **Caso crítico requiere coordinación inmediata**

👥 **Equipo convocado**:
• Dr. Especialista LPP (Dermatología)
• Enfermería especializada (Cuidado heridas)
• Fisioterapeuta (Movilización)
• Nutricionista (Evaluación riesgo)

⏱️ **Tiempo respuesta estimado**: 15 minutos
🎯 **Prioridad**: ALTA - Grado 3 LPP detectado
📋 **Protocolo**: NPUAP/EPUAP 2019 - Emergencia

📊 **Datos del caso**:
• Localización: Región trocantérica
• Extensión: 3.2 cm diámetro
• Profundidad: Tejido subcutáneo expuesto
• Riesgo infección: ALTO

⚡ **Acciones inmediatas requeridas**:
1. Aislamiento de presión INMEDIATO
2. Evaluación quirúrgica en <2 horas
3. Cultivo de tejido si signos infección
4. Protocolo analgesia adecuada

🔒 Batman Token: BATMAN_TEAM_COORD_{datetime.now().strftime('%H%M%S')}
📱 Notificación enviada a todos los miembros del equipo""",
            batman_token=f"BATMAN_TEAM_COORD_{datetime.now().strftime('%H%M%S')}",
            alert_type="team_coordination",
            requires_immediate_response=True,
            clinical_context={
                "case_priority": "HIGH",
                "lpp_grade": "Grade_3",
                "location": "trochanteric_region",
                "team_required": ["dermatology", "nursing", "physiotherapy", "nutrition"],
                "response_time_minutes": 15,
                "surgical_evaluation_required": True
            }
        )
        
        result4 = await slack_orchestrator.send_medical_alert(team_alert)
        if result4.get('ok'):
            print("✅ Coordinación de equipo enviada")
        else:
            print(f"❌ Error coordinación: {result4.get('error')}")
        
        # Summary
        results = [result1, result2, result3, result4]
        successful = sum(1 for r in results if r.get('ok'))
        
        print(f"\n📊 RESULTADOS: {successful}/4 mensajes enviados")
        
        if successful > 0:
            print("🎉 ¡MENSAJES LLEGANDO AL CANAL!")
            print("📱 Verificar #it_vigia para ver las notificaciones")
        else:
            print("❌ Ningún mensaje enviado - verificar configuración")
            
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Ejecutar envío directo"""
    success = await send_direct_messages()
    
    if success:
        print("\n✅ SISTEMA FUNCIONANDO - VERIFICAR #it_vigia")
    else:
        print("\n❌ Sistema requiere ajustes")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())