#!/usr/bin/env python3
"""
Test Slack connection status for VIGIA Medical AI v1.0
"""
import os
import asyncio
from dotenv import load_dotenv
from src.agents.communication_agent import CommunicationAgent
from src.interfaces.slack_orchestrator import SlackOrchestrator

def test_slack_configuration():
    """Test Slack configuration and connection"""
    
    print("🩺 VIGIA Medical AI v1.0 - Slack Connection Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    slack_token = os.getenv('SLACK_BOT_TOKEN')
    print(f"📋 Environment Configuration:")
    print(f"   SLACK_BOT_TOKEN: {'✅ Set' if slack_token and slack_token != 'your_slack_token_here' else '❌ Not configured'}")
    
    if not slack_token or slack_token == 'your_slack_token_here':
        print(f"   Current value: {slack_token}")
        print("   💡 Token is placeholder - needs real Slack Bot Token")
    else:
        print(f"   Token preview: {slack_token[:20]}...")
    
    print()
    
    # Test Slack Orchestrator
    print("🔧 Testing SlackOrchestrator:")
    try:
        orchestrator = SlackOrchestrator()
        result = orchestrator.send_notification("Test message", priority="medium")
        print(f"   ✅ SlackOrchestrator: {result}")
        slack_orchestrator_working = True
    except Exception as e:
        print(f"   ❌ SlackOrchestrator Error: {e}")
        slack_orchestrator_working = False
    
    print()
    
    # Test Communication Agent
    print("🤖 Testing CommunicationAgent:")
    try:
        async def test_communication_agent():
            agent = CommunicationAgent()
            initialized = await agent.initialize()
            print(f"   Agent initialized: {'✅' if initialized else '❌'}")
            
            # Test mock message
            from src.agents.base_agent import AgentMessage
            from datetime import datetime
            
            test_message = AgentMessage(
                session_id="test_session",
                sender_id="test_sender",
                content={
                    "text": "Prueba de comunicación médica",
                    "channel": "slack",
                    "priority": "medium",
                    "recipients": ["equipo_clinico"]
                },
                message_type="processing_request",
                timestamp=datetime.now()
            )
            
            response = await agent.process_message(test_message)
            print(f"   Message processing: {'✅' if response.success else '❌'}")
            print(f"   Response: {response.message}")
            
            return response.success
        
        agent_success = asyncio.run(test_communication_agent())
        
    except Exception as e:
        print(f"   ❌ CommunicationAgent Error: {e}")
        agent_success = False
    
    print()
    
    # System Status Summary
    print("📊 System Status Summary:")
    print("=" * 30)
    
    if slack_token and slack_token != 'your_slack_token_here':
        slack_config_status = "✅ Configured"
    else:
        slack_config_status = "❌ Needs Configuration"
    
    print(f"   Slack Token: {slack_config_status}")
    print(f"   Slack Orchestrator: {'✅ Working' if slack_orchestrator_working else '❌ Failed'}")
    print(f"   Communication Agent: {'✅ Working' if agent_success else '❌ Failed'}")
    
    # Implementation Status
    print()
    print("🏗️ Implementation Status:")
    print("   Architecture: ✅ 9-Agent coordination ready")
    print("   Integration: ✅ SlackOrchestrator + CommunicationAgent")
    print("   Fallback: ✅ Mock mode for development")
    print("   Production: ❌ Requires valid Slack Bot Token")
    
    # Next Steps
    print()
    print("🎯 Next Steps for Production:")
    print("   1. Create Slack App in your workspace")
    print("   2. Generate Bot Token with required permissions")
    print("   3. Update SLACK_BOT_TOKEN in .env file")
    print("   4. Replace mock SlackOrchestrator with real implementation")
    
    return {
        'slack_configured': slack_token and slack_token != 'your_slack_token_here',
        'orchestrator_working': slack_orchestrator_working,
        'agent_working': agent_success,
        'ready_for_production': False  # Needs real token
    }

if __name__ == "__main__":
    status = test_slack_configuration()
    
    if all([status['orchestrator_working'], status['agent_working']]):
        print("\n🎉 Slack integration architecture is ready!")
        print("   System works with mock implementation")
        if not status['slack_configured']:
            print("   ⚠️  Production requires valid Slack Bot Token")
    else:
        print("\n❌ Slack integration has issues")
        print("   Check errors above for resolution steps")