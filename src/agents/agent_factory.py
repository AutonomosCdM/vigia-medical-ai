"""
Agent Factory for Vigia Medical System
======================================

Comprehensive agent factory and orchestration system for creating,
managing, and coordinating all 9 specialized medical agents in the
VIGIA Medical AI system.

Key Features:
- Complete 9-agent instantiation and configuration
- ADK integration with A2A communication setup
- Agent health monitoring and status tracking
- Production-ready agent orchestration
- Batman tokenization compliance
- Medical workflow coordination
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

# Import all specialized agents
from src.agents.image_analysis_agent import ImageAnalysisAgent
from src.agents.clinical_assessment_agent import ClinicalAssessmentAgent
from src.agents.protocol_agent import ProtocolAgent
from src.agents.communication_agent import CommunicationAgent
from src.agents.workflow_orchestration_agent import WorkflowOrchestrationAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.monai_review_agent import MonaiReviewAgent
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.voice_analysis_agent import VoiceAnalysisAgent

# ADK agent removed - using standard communication agent

# Import orchestrator
from src.agents.master_medical_orchestrator import MasterMedicalOrchestrator

# Import base classes
from src.agents.base_agent import BaseAgent, AgentMessage, AgentResponse

# Utilities
from src.utils.audit_service import AuditService, AuditEventType
from src.utils.secure_logger import SecureLogger

logger = SecureLogger("agent_factory")


class AgentType(Enum):
    """Types of agents in the Vigia system"""
    IMAGE_ANALYSIS = "image_analysis"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    PROTOCOL = "protocol"
    COMMUNICATION = "communication"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    RISK_ASSESSMENT = "risk_assessment"
    MONAI_REVIEW = "monai_review"
    DIAGNOSTIC = "diagnostic"
    VOICE_ANALYSIS = "voice_analysis"
    MASTER_ORCHESTRATOR = "master_orchestrator"


class AgentStatus(Enum):
    """Agent status states"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentInstance:
    """Agent instance information"""
    agent_id: str
    agent_type: str
    instance: Union[BaseAgent, Any]  # Can be BaseAgent or ADK agent
    status: str
    created_at: datetime
    last_health_check: datetime
    capabilities: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'last_health_check': self.last_health_check.isoformat(),
            'capabilities': self.capabilities,
            'metadata': self.metadata,
            'is_adk_agent': hasattr(self.instance, 'model'),  # ADK agents have model attribute
            'is_base_agent': isinstance(self.instance, BaseAgent)
        }


class VigiaAgentFactory:
    """
    Comprehensive agent factory for the Vigia Medical System.
    
    Capabilities:
    - Create and manage all 9 specialized medical agents
    - ADK integration with A2A communication
    - Agent health monitoring and status tracking
    - Production-ready orchestration setup
    - Agent discovery and capability mapping
    - Medical workflow coordination
    """
    
    def __init__(self):
        """Initialize the agent factory"""
        self.audit_service = AuditService()
        
        # Agent registry
        self.agents: Dict[str, AgentInstance] = {}
        self.agent_types_registry: Dict[str, Type] = {}
        
        # Agent creation mappings
        self.agent_creators = {
            AgentType.IMAGE_ANALYSIS.value: self._create_image_analysis_agent,
            AgentType.CLINICAL_ASSESSMENT.value: self._create_clinical_assessment_agent,
            AgentType.PROTOCOL.value: self._create_protocol_agent,
            AgentType.COMMUNICATION.value: self._create_communication_agent,
            AgentType.WORKFLOW_ORCHESTRATION.value: self._create_workflow_orchestration_agent,
            AgentType.RISK_ASSESSMENT.value: self._create_risk_assessment_agent,
            AgentType.MONAI_REVIEW.value: self._create_monai_review_agent,
            AgentType.DIAGNOSTIC.value: self._create_diagnostic_agent,
            AgentType.VOICE_ANALYSIS.value: self._create_voice_analysis_agent,
            AgentType.MASTER_ORCHESTRATOR.value: self._create_master_orchestrator
        }
        
        # Factory statistics
        self.stats = {
            'agents_created': 0,
            'agents_active': 0,
            'agents_failed': 0,
            'last_health_check': None,
            'orchestrator_active': False
        }
        
        # Register agent types
        self._register_agent_types()
    
    def _register_agent_types(self):
        """Register all agent types for factory creation"""
        self.agent_types_registry = {
            AgentType.IMAGE_ANALYSIS.value: ImageAnalysisAgent,
            AgentType.CLINICAL_ASSESSMENT.value: ClinicalAssessmentAgent,
            AgentType.PROTOCOL.value: ProtocolAgent,
            AgentType.COMMUNICATION.value: CommunicationAgent,
            AgentType.WORKFLOW_ORCHESTRATION.value: WorkflowOrchestrationAgent,
            AgentType.RISK_ASSESSMENT.value: RiskAssessmentAgent,
            AgentType.MONAI_REVIEW.value: MonaiReviewAgent,
            AgentType.DIAGNOSTIC.value: DiagnosticAgent,
            AgentType.VOICE_ANALYSIS.value: VoiceAnalysisAgent
        }
    
    async def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentInstance:
        """
        Create a single agent instance.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional custom agent ID
            config: Optional agent configuration
            
        Returns:
            Created agent instance
        """
        try:
            if agent_type not in self.agent_creators:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Generate agent ID if not provided
            if not agent_id:
                agent_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create agent using appropriate creator
            creator = self.agent_creators[agent_type]
            agent_instance = await creator(agent_id, config or {})
            
            # Create agent wrapper
            agent_wrapper = AgentInstance(
                agent_id=agent_id,
                agent_type=agent_type,
                instance=agent_instance,
                status=AgentStatus.INITIALIZING.value,
                created_at=datetime.now(timezone.utc),
                last_health_check=datetime.now(timezone.utc),
                capabilities=self._get_agent_capabilities(agent_instance),
                metadata={
                    'config': config or {},
                    'creation_time': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Initialize agent
            if hasattr(agent_instance, 'initialize'):
                await agent_instance.initialize()
            
            # Update status
            agent_wrapper.status = AgentStatus.HEALTHY.value
            
            # Register agent
            self.agents[agent_id] = agent_wrapper
            self.stats['agents_created'] += 1
            self.stats['agents_active'] += 1
            
            # Log creation
            await self._log_agent_event(
                agent_id,
                "agent_created",
                {
                    "agent_type": agent_type,
                    "capabilities_count": len(agent_wrapper.capabilities),
                    "config_provided": bool(config)
                }
            )
            
            logger.info(f"Created agent {agent_id} of type {agent_type}")
            return agent_wrapper
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}")
            self.stats['agents_failed'] += 1
            raise
    
    async def create_complete_medical_system(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, AgentInstance]:
        """
        Create complete 9-agent medical system with orchestrator.
        
        Args:
            config: Optional system-wide configuration
            
        Returns:
            Dictionary of all created agents
        """
        try:
            logger.info("Creating complete Vigia Medical AI system...")
            
            # Default configuration
            system_config = config or {}
            
            # Create all core agents in parallel
            agent_creation_tasks = []
            
            for agent_type in [
                AgentType.IMAGE_ANALYSIS.value,
                AgentType.CLINICAL_ASSESSMENT.value,
                AgentType.PROTOCOL.value,
                AgentType.COMMUNICATION.value,
                AgentType.WORKFLOW_ORCHESTRATION.value,
                AgentType.RISK_ASSESSMENT.value,
                AgentType.MONAI_REVIEW.value,
                AgentType.DIAGNOSTIC.value,
                AgentType.VOICE_ANALYSIS.value
            ]:
                task = self.create_agent(
                    agent_type=agent_type,
                    config=system_config.get(agent_type, {})
                )
                agent_creation_tasks.append(task)
            
            # Create all agents concurrently
            created_agents = await asyncio.gather(*agent_creation_tasks, return_exceptions=True)
            
            # Process results
            successful_agents = {}
            failed_agents = []
            
            for i, result in enumerate(created_agents):
                if isinstance(result, Exception):
                    agent_type = list(AgentType)[i].value
                    failed_agents.append((agent_type, str(result)))
                    logger.error(f"Failed to create {agent_type}: {result}")
                else:
                    successful_agents[result.agent_id] = result
            
            # Create master orchestrator if all core agents succeeded
            if len(failed_agents) == 0:
                try:
                    orchestrator = await self.create_agent(
                        agent_type=AgentType.MASTER_ORCHESTRATOR.value,
                        config=system_config.get('orchestrator', {})
                    )
                    successful_agents[orchestrator.agent_id] = orchestrator
                    self.stats['orchestrator_active'] = True
                    
                    logger.info("Master orchestrator created successfully")
                except Exception as e:
                    logger.error(f"Failed to create master orchestrator: {e}")
                    failed_agents.append(('master_orchestrator', str(e)))
            
            # Setup A2A communication between agents
            await self._setup_agent_communication(successful_agents)
            
            # Log system creation
            await self._log_agent_event(
                "system",
                "medical_system_created",
                {
                    "successful_agents": len(successful_agents),
                    "failed_agents": len(failed_agents),
                    "orchestrator_created": self.stats['orchestrator_active'],
                    "failed_agent_types": [agent_type for agent_type, _ in failed_agents]
                }
            )
            
            logger.info(f"Medical system creation complete: {len(successful_agents)} agents created, {len(failed_agents)} failed")
            
            if failed_agents:
                logger.warning(f"Failed agent types: {[at for at, _ in failed_agents]}")
            
            return successful_agents
            
        except Exception as e:
            logger.error(f"Failed to create complete medical system: {e}")
            raise
    
    async def _create_image_analysis_agent(self, agent_id: str, config: Dict[str, Any]) -> ImageAnalysisAgent:
        """Create Image Analysis Agent"""
        return ImageAnalysisAgent(agent_id=agent_id)
    
    async def _create_clinical_assessment_agent(self, agent_id: str, config: Dict[str, Any]) -> ClinicalAssessmentAgent:
        """Create Clinical Assessment Agent"""
        return ClinicalAssessmentAgent(agent_id=agent_id)
    
    async def _create_protocol_agent(self, agent_id: str, config: Dict[str, Any]) -> ProtocolAgent:
        """Create Protocol Agent"""
        return ProtocolAgent(agent_id=agent_id)
    
    async def _create_communication_agent(self, agent_id: str, config: Dict[str, Any]) -> CommunicationAgent:
        """Create Communication Agent (can be ADK or base)"""
        use_adk = config.get('use_adk', True)
        
        if use_adk:
            # Create ADK Communication Agent
            return CommunicationAgentADKFactory.create_agent()
        else:
            # Create base Communication Agent
            return CommunicationAgent(agent_id=agent_id)
    
    async def _create_workflow_orchestration_agent(self, agent_id: str, config: Dict[str, Any]) -> WorkflowOrchestrationAgent:
        """Create Workflow Orchestration Agent"""
        return WorkflowOrchestrationAgent(agent_id=agent_id)
    
    async def _create_risk_assessment_agent(self, agent_id: str, config: Dict[str, Any]) -> RiskAssessmentAgent:
        """Create Risk Assessment Agent"""
        return RiskAssessmentAgent(agent_id=agent_id)
    
    async def _create_monai_review_agent(self, agent_id: str, config: Dict[str, Any]) -> MonaiReviewAgent:
        """Create MONAI Review Agent"""
        return MonaiReviewAgent(agent_id=agent_id)
    
    async def _create_diagnostic_agent(self, agent_id: str, config: Dict[str, Any]) -> DiagnosticAgent:
        """Create Diagnostic Agent"""
        return DiagnosticAgent(agent_id=agent_id)
    
    async def _create_voice_analysis_agent(self, agent_id: str, config: Dict[str, Any]) -> VoiceAnalysisAgent:
        """Create Voice Analysis Agent"""
        return VoiceAnalysisAgent(agent_id=agent_id)
    
    async def _create_master_orchestrator(self, agent_id: str, config: Dict[str, Any]) -> MasterMedicalOrchestrator:
        """Create Master Medical Orchestrator"""
        return MasterMedicalOrchestrator()
    
    def _get_agent_capabilities(self, agent_instance: Any) -> List[str]:
        """Get agent capabilities"""
        try:
            if hasattr(agent_instance, 'get_capabilities'):
                return agent_instance.get_capabilities()
            elif hasattr(agent_instance, 'tools'):
                # ADK agent
                return [tool.__name__ for tool in agent_instance.tools]
            else:
                return ['basic_agent_capabilities']
        except Exception as e:
            logger.warning(f"Failed to get agent capabilities: {e}")
            return ['unknown_capabilities']
    
    async def _setup_agent_communication(self, agents: Dict[str, AgentInstance]):
        """Setup A2A communication between agents"""
        try:
            logger.info("Setting up A2A communication between agents...")
            
            # Create agent registry for communication
            agent_registry = {}
            for agent_id, agent_wrapper in agents.items():
                agent_registry[agent_id] = {
                    'agent_type': agent_wrapper.agent_type,
                    'capabilities': agent_wrapper.capabilities,
                    'instance': agent_wrapper.instance
                }
            
            # Configure each agent with registry of other agents
            for agent_id, agent_wrapper in agents.items():
                if hasattr(agent_wrapper.instance, 'set_agent_registry'):
                    other_agents = {k: v for k, v in agent_registry.items() if k != agent_id}
                    await agent_wrapper.instance.set_agent_registry(other_agents)
            
            logger.info(f"A2A communication setup complete for {len(agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to setup agent communication: {e}")
    
    async def health_check_all_agents(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        try:
            health_results = {}
            
            for agent_id, agent_wrapper in self.agents.items():
                try:
                    if hasattr(agent_wrapper.instance, 'get_health_status'):
                        health_status = agent_wrapper.instance.get_health_status()
                    else:
                        health_status = {'status': 'unknown', 'message': 'Health check not supported'}
                    
                    agent_wrapper.status = AgentStatus.HEALTHY.value
                    agent_wrapper.last_health_check = datetime.now(timezone.utc)
                    
                    health_results[agent_id] = {
                        'agent_type': agent_wrapper.agent_type,
                        'status': 'healthy',
                        'health_data': health_status,
                        'last_check': agent_wrapper.last_health_check.isoformat()
                    }
                    
                except Exception as e:
                    agent_wrapper.status = AgentStatus.ERROR.value
                    health_results[agent_id] = {
                        'agent_type': agent_wrapper.agent_type,
                        'status': 'error',
                        'error': str(e),
                        'last_check': datetime.now(timezone.utc).isoformat()
                    }
            
            # Update factory statistics
            healthy_agents = sum(1 for r in health_results.values() if r['status'] == 'healthy')
            self.stats['agents_active'] = healthy_agents
            self.stats['last_health_check'] = datetime.now(timezone.utc).isoformat()
            
            return {
                'total_agents': len(self.agents),
                'healthy_agents': healthy_agents,
                'agent_details': health_results,
                'factory_stats': self.stats,
                'health_check_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[AgentInstance]:
        """Get all agents of a specific type"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]
    
    def list_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all agents with their information"""
        return {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()}
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get complete system capabilities"""
        all_capabilities = set()
        agent_type_capabilities = {}
        
        for agent_wrapper in self.agents.values():
            all_capabilities.update(agent_wrapper.capabilities)
            agent_type_capabilities[agent_wrapper.agent_type] = agent_wrapper.capabilities
        
        return {
            'total_unique_capabilities': len(all_capabilities),
            'all_capabilities': sorted(list(all_capabilities)),
            'agents_count': len(self.agents),
            'agent_types': list(agent_type_capabilities.keys()),
            'capabilities_by_agent_type': agent_type_capabilities,
            'orchestrator_active': self.stats['orchestrator_active']
        }
    
    async def shutdown_all_agents(self):
        """Gracefully shutdown all agents"""
        try:
            logger.info("Shutting down all agents...")
            
            shutdown_tasks = []
            for agent_id, agent_wrapper in self.agents.items():
                if hasattr(agent_wrapper.instance, 'shutdown'):
                    task = agent_wrapper.instance.shutdown()
                    shutdown_tasks.append(task)
            
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Clear agent registry
            self.agents.clear()
            self.stats['agents_active'] = 0
            self.stats['orchestrator_active'] = False
            
            logger.info("All agents shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
    
    async def _log_agent_event(
        self,
        agent_id: str,
        event_type: str,
        details: Dict[str, Any]
    ):
        """Log agent event for audit trail"""
        try:
            await self.audit_service.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                component="agent_factory",
                action=event_type,
                details={
                    "agent_id": agent_id,
                    "factory_stats": self.stats,
                    **details
                }
            )
        except Exception as e:
            logger.warning(f"Agent event logging failed: {e}")


# Factory singleton
_factory_instance: Optional[VigiaAgentFactory] = None


def get_agent_factory() -> VigiaAgentFactory:
    """Get singleton agent factory instance"""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = VigiaAgentFactory()
    return _factory_instance


async def create_complete_vigia_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, AgentInstance]:
    """
    Create complete Vigia Medical AI system with all 9 agents.
    
    Args:
        config: Optional system configuration
        
    Returns:
        Dictionary of all created agents
    """
    factory = get_agent_factory()
    return await factory.create_complete_medical_system(config)


async def quick_agent_health_check() -> Dict[str, Any]:
    """Perform quick health check on all agents"""
    factory = get_agent_factory()
    return await factory.health_check_all_agents()


# Export for use
__all__ = [
    'VigiaAgentFactory',
    'AgentInstance',
    'AgentType',
    'AgentStatus',
    'get_agent_factory',
    'create_complete_vigia_system',
    'quick_agent_health_check'
]