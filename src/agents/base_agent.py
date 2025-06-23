"""
Base Agent - Foundation for ADK Medical Agents
==============================================

Provides base classes and structures for Agent-to-Agent (A2A) communication
and standardized medical agent messaging in the Vigia system.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid


class MessageType(Enum):
    """Message types for A2A communication"""
    PROCESSING_REQUEST = "processing_request"
    PROCESSING_RESPONSE = "processing_response"
    STATUS_UPDATE = "status_update"
    ERROR_NOTIFICATION = "error_notification"
    HEALTH_CHECK = "health_check"


class AgentCapability(Enum):
    """Agent capabilities for medical processing"""
    IMAGE_ANALYSIS = "image_analysis"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    PROTOCOL_CONSULTATION = "protocol_consultation"
    MEDICAL_COMMUNICATION = "medical_communication"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    LPP_DETECTION = "lpp_detection"
    EVIDENCE_BASED_DECISIONS = "evidence_based_decisions"
    HIPAA_COMPLIANCE = "hipaa_compliance"


@dataclass
class AgentMessage:
    """Standardized message structure for A2A communication"""
    session_id: str
    sender_id: str
    content: Dict[str, Any]
    message_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    message_id: str = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResponse:
    """Standardized response structure for A2A communication"""
    success: bool
    content: Dict[str, Any]
    message: str
    timestamp: datetime
    requires_human_review: bool = False
    next_actions: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    response_id: str = None
    
    def __post_init__(self):
        if self.response_id is None:
            self.response_id = str(uuid.uuid4())
        if self.next_actions is None:
            self.next_actions = []
        if self.metadata is None:
            self.metadata = {}


class BaseAgent:
    """Base class for all medical agents in the Vigia system"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.initialized = False
        self.stats = {
            'messages_processed': 0,
            'errors_encountered': 0,
            'avg_processing_time': 0.0
        }
    
    async def initialize(self):
        """Initialize the agent"""
        self.initialized = True
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming A2A message"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Update stats
            self.stats['messages_processed'] += 1
            
            # Process the message (to be implemented by subclasses)
            result = await self._process_message_impl(message)
            
            return AgentResponse(
                success=True,
                content=result,
                message="Message processed successfully",
                timestamp=datetime.now(timezone.utc),
                metadata={'agent_id': self.agent_id, 'agent_type': self.agent_type}
            )
            
        except Exception as e:
            self.stats['errors_encountered'] += 1
            return AgentResponse(
                success=False,
                content={'error': str(e)},
                message=f"Error processing message: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                requires_human_review=True,
                metadata={'agent_id': self.agent_id, 'agent_type': self.agent_type}
            )
    
    async def _process_message_impl(self, message: AgentMessage) -> Dict[str, Any]:
        """Implementation-specific message processing (to be overridden)"""
        raise NotImplementedError("Subclasses must implement _process_message_impl")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'initialized': self.initialized,
            'stats': self.stats,
            'uptime': datetime.now(timezone.utc).isoformat()
        }


# Export classes for use
__all__ = [
    'MessageType',
    'AgentCapability',
    'AgentMessage', 
    'AgentResponse',
    'BaseAgent'
]