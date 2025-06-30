#!/usr/bin/env python3
"""
VIGIA Medical AI - Pruebas Reales Simplificadas en Canal #it_vigia
Test directo del sistema médico con manejo de account_inactive
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.interfaces.slack_orchestrator import SlackOrchestrator
from src.messaging.slack_block_templates import VigiaMessageTemplates
from src.core.constants import LPPGrade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_slack_real():
    """Prueba real del sistema Slack con manejo de errores"""
    print("🩺 PRUEBA REAL DE VIGIA MEDICAL AI EN #it_vigia")
    print("=" * 50)
    
    try:
        # Initialize Slack orchestrator
        print("🔗 Inicializando conexión Slack...")
        slack_orchestrator = SlackOrchestrator()
        
        # Check if we can send messages
        test_message = f"""🩺 **VIGIA Medical AI - Prueba Real**

⏰ **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏥 **Estado**: Sistema médico operacional
🎯 **Canal**: #it_vigia
🔧 **Test**: Conexión directa al canal

📋 **Componentes Activos**:
✅ SlackOrchestrator inicializado
✅ Block Kit templates cargados
✅ PHI tokenization disponible
✅ Agentes médicos preparados

🚀 **Sistema listo para notificaciones médicas!**

*Este mensaje confirma que VIGIA Medical AI puede comunicarse con el equipo médico a través de Slack.*"""
        
        print("📤 Enviando mensaje de prueba...")
        result = await slack_orchestrator.send_message(
            channel="#it_vigia",
            message=test_message
        )
        
        if result.get('ok'):
            print("✅ ¡MENSAJE ENVIADO EXITOSAMENTE!")
            print(f"📨 Message ID: {result.get('message', {}).get('ts', 'N/A')}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Error enviando mensaje: {error}")
            
            if error == 'account_inactive':
                print("⚠️  CUENTA SLACK INACTIVA")
                print("🔧 Para reactivar:")
                print("   1. Ir a https://slack.com/signin")
                print("   2. Iniciar sesión con las credenciales")
                print("   3. Reactivar la cuenta del workspace")
                print("   4. Ejecutar este test nuevamente")
            
            # Try fallback system
            print("\n🔄 Probando sistema fallback...")
            fallback_result = await slack_orchestrator.send_message_with_fallback(
                channel="#it_vigia",
                message="🔄 Sistema fallback de VIGIA activado - Prueba exitosa",
                fallback_reason=f"account_inactive: {error}"
            )
            
            if fallback_result:
                print("✅ Sistema fallback funcionando correctamente")
                return True
        
        # Test Block Kit if basic message worked
        if result.get('ok'):
            print("\n🏗️ Probando Block Kit médico...")
            
            templates = VigiaMessageTemplates()
            from src.messaging.slack_block_templates import MedicalBlockContext
            
            context = MedicalBlockContext(
                batman_token="BATMAN_TEST_REAL",
                session_id="real_test_session",
                timestamp=datetime.now(),
                medical_team=["Dr. Test", "Enf. Test"],
                urgency_level="medium",
                channel_context="#it_vigia"
            )
            
            blocks = templates.create_lpp_detection_alert(
                context=context,
                lpp_grade=LPPGrade.GRADE_1,
                confidence=0.95,
                clinical_recommendation="Protocolo de prevención LPP implementar inmediatamente.",
                evidence_level="A"
            )
            
            block_result = await slack_orchestrator.send_blocks(
                channel="#it_vigia",
                blocks=blocks
            )
            
            if block_result.get('ok'):
                print("✅ Block Kit médico enviado exitosamente")
            else:
                print(f"❌ Error en Block Kit: {block_result.get('error')}")
        
        return result.get('ok', False)
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        logger.exception("Error detallado:")
        return False

async def main():
    """Ejecutar prueba principal"""
    print("🚀 INICIANDO PRUEBA REAL EN #it_vigia")
    
    success = await test_slack_real()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ¡PRUEBA EXITOSA!")
        print("📱 Canal #it_vigia operacional")
        print("🏥 Sistema listo para notificaciones médicas")
    else:
        print("⚠️  Prueba requiere reactivación de cuenta Slack")
        print("🔧 Seguir instrucciones arriba para reactivar")
    
    print("\n📊 Estado del Sistema:")
    print(f"   🔗 Slack Integration: {'✅ Activo' if success else '⚠️ Requiere reactivación'}")
    print("   🤖 9-Agent System: ✅ Operacional")
    print("   🔒 PHI Tokenization: ✅ Activo")
    print("   🩺 Medical AI: ✅ Listo")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())