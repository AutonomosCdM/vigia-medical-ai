#!/usr/bin/env python3
"""
VIGIA Medical AI - AgentOps Setup Script
======================================

Configures AgentOps integration for the 9-agent medical system with HIPAA compliance.
Validates API key, tests connectivity, and verifies telemetry configuration.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.monitoring.agentops_client import AgentOpsClient
from src.monitoring.medical_telemetry import MedicalTelemetry
from src.utils.secure_logger import SecureLogger

logger = SecureLogger("agentops_setup")

class AgentOpsSetup:
    """Setup and validation for AgentOps integration"""
    
    def __init__(self):
        self.env_file = Path(__file__).parent.parent / ".env"
        self.agentops_client: Optional[AgentOpsClient] = None
        
    def print_banner(self):
        """Print setup banner"""
        print("=" * 60)
        print("🏥 VIGIA Medical AI - AgentOps Integration Setup")
        print("=" * 60)
        print("Setting up monitoring for 9-agent medical system")
        print("HIPAA-compliant telemetry with Batman token protection")
        print()
    
    def check_api_key(self, api_key: str = None) -> bool:
        """Validate AgentOps API key format"""
        
        # Get API key from parameter, environment, or prompt
        if not api_key:
            api_key = os.getenv('AGENTOPS_API_KEY')
        
        if not api_key:
            print("🔑 AgentOps API Key Required")
            print("Get your API key from: https://app.agentops.ai/settings/api-keys")
            api_key = input("Enter your AgentOps API key: ").strip()
        
        # Validate format (should start with sk-ao-)
        if not api_key.startswith('sk-ao-'):
            print("❌ Invalid API key format. Should start with 'sk-ao-'")
            return False
        
        # Update environment file
        self.update_env_file(api_key)
        print(f"✅ API key validated and saved to {self.env_file}")
        return True
    
    def update_env_file(self, api_key: str):
        """Update .env file with AgentOps configuration"""
        
        # Read existing .env content
        env_content = ""
        if self.env_file.exists():
            env_content = self.env_file.read_text()
        
        # Update or add AgentOps configuration
        lines = env_content.split('\n')
        updated_lines = []
        agentops_section_found = False
        
        for line in lines:
            if line.startswith('AGENTOPS_API_KEY='):
                updated_lines.append(f'AGENTOPS_API_KEY={api_key}')
                agentops_section_found = True
            elif line.startswith('# Monitoring & Telemetry'):
                agentops_section_found = True
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Add AgentOps section if not found
        if not agentops_section_found:
            updated_lines.extend([
                "",
                "# Monitoring & Telemetry (AgentOps Integration)",
                "# Get your API key from: https://app.agentops.ai/settings/api-keys",
                f"AGENTOPS_API_KEY={api_key}",
                "TELEMETRY_ENABLED=true",
                "PERFORMANCE_MONITORING=true",
                "AGENTOPS_ENVIRONMENT=production",
                "AGENTOPS_TAGS=medical-ai,vigia,hipaa-compliant,pressure-injury-detection"
            ])
        
        self.env_file.write_text('\n'.join(updated_lines))
    
    async def test_connectivity(self) -> bool:
        """Test AgentOps connectivity and initialization"""
        
        print("\n🔗 Testing AgentOps connectivity...")
        
        try:
            # Initialize AgentOps client
            self.agentops_client = AgentOpsClient()
            
            if not self.agentops_client.is_initialized:
                print("❌ AgentOps client failed to initialize")
                print("Check your API key and internet connectivity")
                return False
            
            print("✅ AgentOps client initialized successfully")
            
            # Test medical telemetry
            telemetry = MedicalTelemetry(
                agent_id="setup_test_agent",
                agent_type="setup_validation"
            )
            
            # Start a test session
            session_id = await telemetry.start_medical_session(
                batman_token="BATMAN_SETUP_TEST_001",
                case_metadata={"test": "setup_validation", "system": "vigia"}
            )
            
            print("✅ Medical telemetry session started successfully")
            
            # Track a test event
            await telemetry.track_agent_interaction(
                session_id=session_id,
                interaction_type="setup_test",
                interaction_data={"test": "connectivity", "status": "success"}
            )
            
            print("✅ Test telemetry event tracked successfully")
            
            # End test session
            await telemetry.end_medical_session(
                session_id=session_id,
                outcome="setup_test_completed"
            )
            
            print("✅ Test session completed successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ AgentOps connectivity test failed: {e}")
            return False
    
    def validate_agent_integration(self) -> Dict[str, bool]:
        """Validate all 9 medical agents have telemetry integration"""
        
        print("\n🤖 Validating medical agent telemetry integration...")
        
        agents_to_check = [
            "master_medical_orchestrator.py",
            "image_analysis_agent.py", 
            "voice_analysis_agent.py",
            "clinical_assessment_agent.py",
            "risk_assessment_agent.py",
            "diagnostic_agent.py",
            "protocol_agent.py",
            "communication_agent.py",
            "monai_review_agent.py"
        ]
        
        agents_dir = Path(__file__).parent.parent / "src" / "agents"
        validation_results = {}
        
        for agent_file in agents_to_check:
            agent_path = agents_dir / agent_file
            if not agent_path.exists():
                validation_results[agent_file] = False
                print(f"❌ {agent_file}: File not found")
                continue
            
            # Check for telemetry imports
            content = agent_path.read_text()
            has_telemetry = all([
                "from src.monitoring.medical_telemetry import MedicalTelemetry" in content,
                "self.telemetry = MedicalTelemetry(" in content
            ])
            
            validation_results[agent_file] = has_telemetry
            status = "✅" if has_telemetry else "❌"
            print(f"{status} {agent_file}: {'Telemetry integrated' if has_telemetry else 'Missing telemetry'}")
        
        return validation_results
    
    def print_setup_summary(self, connectivity_test: bool, agent_validation: Dict[str, bool]):
        """Print setup completion summary"""
        
        print("\n" + "=" * 60)
        print("🏥 VIGIA Medical AI - AgentOps Setup Summary")
        print("=" * 60)
        
        # Connectivity status
        conn_status = "✅ Connected" if connectivity_test else "❌ Failed"
        print(f"AgentOps Connectivity: {conn_status}")
        
        # Agent integration status
        integrated_count = sum(agent_validation.values())
        total_agents = len(agent_validation)
        print(f"Agent Integration: {integrated_count}/{total_agents} agents configured")
        
        if connectivity_test and integrated_count == total_agents:
            print("\n🎉 AgentOps setup completed successfully!")
            print("All 9 medical agents are now monitored with HIPAA compliance.")
            
            # Print next steps
            print("\n📋 Next Steps:")
            print("1. Deploy to AWS: cd infrastructure && python deploy_vigia.py")
            print("2. Monitor agents: https://vigia.autonomos.dev/monitoring/agentops/status")
            print("3. View dashboard: https://app.agentops.ai")
            
        else:
            print("\n⚠️  Setup completed with issues")
            if not connectivity_test:
                print("- Fix AgentOps connectivity before deployment")
            if integrated_count < total_agents:
                print(f"- {total_agents - integrated_count} agents need telemetry integration")
        
        print("=" * 60)
    
    async def run_setup(self, api_key: str = None, skip_validation: bool = False):
        """Run complete AgentOps setup process"""
        
        self.print_banner()
        
        # Step 1: Validate API key
        if not self.check_api_key(api_key):
            return False
        
        # Step 2: Test connectivity
        connectivity_test = await self.test_connectivity()
        
        # Step 3: Validate agent integration
        agent_validation = {} if skip_validation else self.validate_agent_integration()
        
        # Step 4: Print summary
        self.print_setup_summary(connectivity_test, agent_validation)
        
        return connectivity_test and (skip_validation or all(agent_validation.values()))

async def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(description="Setup AgentOps for VIGIA Medical AI")
    parser.add_argument("--api-key", help="AgentOps API key")
    parser.add_argument("--skip-validation", action="store_true", 
                       help="Skip agent telemetry validation")
    
    args = parser.parse_args()
    
    setup = AgentOpsSetup()
    success = await setup.run_setup(
        api_key=args.api_key,
        skip_validation=args.skip_validation
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())