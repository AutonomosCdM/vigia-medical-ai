"""
A2A Base Infrastructure - Mock implementation for compatibility
==============================================================
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentCard:
    """Agent card for A2A registration"""
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: list = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

class A2AServer:
    """Mock A2A server for compatibility"""
    
    def __init__(self):
        self.registered_agents = {}
        logger.info("A2AServer initialized (mock mode)")
    
    def register_agent(self, agent_card: AgentCard):
        """Register agent"""
        self.registered_agents[agent_card.agent_id] = agent_card
        logger.info(f"Registered agent: {agent_card.agent_id} (mock)")

__all__ = ['AgentCard', 'A2AServer']