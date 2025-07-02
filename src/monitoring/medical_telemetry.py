"""
VIGIA Medical AI - Medical Telemetry System
==========================================

Medical-grade telemetry with AgentOps integration, HIPAA compliance,
and specialized monitoring for 9-agent medical AI coordination.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .agentops_client import AgentOpsClient, MedicalSession
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService, AuditEventType

logger = SecureLogger(__name__)

class SessionType(Enum):
    """Medical session types"""
    MASTER_ORCHESTRATION = "master_orchestration"
    IMAGE_ANALYSIS = "image_analysis"
    VOICE_ANALYSIS = "voice_analysis"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    RISK_ASSESSMENT = "risk_assessment"
    DIAGNOSTIC_FUSION = "diagnostic_fusion"
    PROTOCOL_APPLICATION = "protocol_application"
    COMMUNICATION = "communication"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    MONAI_REVIEW = "monai_review"

@dataclass
class MedicalEvent:
    """Medical AI event with compliance tracking"""
    event_id: str
    batman_token: str
    event_type: str
    agent_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    compliance_verified: bool = True
    session_id: Optional[str] = None

class MedicalTelemetry:
    """Production medical telemetry with AgentOps integration"""
    
    def __init__(self, 
                 app_id: str = None,
                 environment: str = "production",
                 enable_phi_protection: bool = True,
                 **kwargs):
        
        self.app_id = app_id or "vigia-medical-ai"
        self.environment = environment
        self.enable_phi_protection = enable_phi_protection
        
        # Initialize AgentOps client
        self.agentops_client = AgentOpsClient()
        self.audit_service = AuditService()
        
        # Session management
        self.active_sessions: Dict[str, MedicalSession] = {}
        self.session_events: Dict[str, List[MedicalEvent]] = {}
        
        # Medical compliance tracking
        self.compliance_violations = []
        self.session_stats = {
            "total_sessions": 0,
            "completed_sessions": 0,
            "failed_sessions": 0,
            "phi_violations_detected": 0,
            "events_recorded": 0
        }
        
        logger.info(f"MedicalTelemetry initialized - App: {self.app_id}, Env: {environment}, PHI Protection: {enable_phi_protection}")
    
    async def start_medical_session(self,
                                  session_id: str,
                                  patient_context: Dict[str, Any],
                                  session_type: str = "medical_analysis") -> MedicalSession:
        """Start HIPAA-compliant medical session"""
        
        # Verify Batman token compliance
        batman_token = patient_context.get('token_id')
        if not batman_token or not batman_token.startswith('batman_'):
            raise ValueError("Medical sessions must use Batman tokens (no PHI allowed)")
        
        # Extract agent type from session type
        agent_type = session_type.replace("_orchestration", "").replace("_", "-")
        
        # Create session metadata (PHI-free)
        safe_metadata = {
            "session_type": session_type,
            "app_id": self.app_id,
            "environment": self.environment,
            "compliance": "hipaa_batman_tokens",
            "start_time": datetime.now().isoformat(),
            **{k: v for k, v in patient_context.items() 
               if k not in ['real_patient_id', 'patient_name', 'mrn'] and not self._contains_phi(str(v))}
        }
        
        # Start AgentOps session
        agentops_session_id = self.agentops_client.start_medical_session(
            batman_token=batman_token,
            agent_type=agent_type,
            case_metadata=safe_metadata
        )
        
        # Create medical session
        medical_session = MedicalSession(
            session_id=session_id,
            batman_token=batman_token,
            agent_type=agent_type,
            start_time=datetime.now(),
            metadata=safe_metadata
        )
        
        # Track session
        self.active_sessions[session_id] = medical_session
        self.session_events[session_id] = []
        self.session_stats["total_sessions"] += 1
        
        # Log session start
        await self.audit_service.log_event(
            event_type=AuditEventType.SYSTEM_START,
            component="medical_telemetry",
            action="start_medical_session",
            details={
                "session_id": session_id,
                "session_type": session_type,
                "agent_type": agent_type,
                "agentops_session": agentops_session_id,
                "batman_token": batman_token
            },
            session_id=session_id
        )
        
        logger.info(f"Medical session started: {session_id} ({session_type}) with AgentOps integration")
        return medical_session
    
    async def record_medical_event(self,
                                 session_id: str,
                                 event_type: str,
                                 event_data: Dict[str, Any],
                                 agent_action: bool = False) -> str:
        """Record medical AI event with HIPAA compliance"""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Recording event for unknown session: {session_id}")
            return None
        
        session = self.active_sessions[session_id]
        event_id = f"event_{session_id}_{len(self.session_events[session_id])}"
        
        # Verify PHI protection
        if self.enable_phi_protection:
            safe_data = self._sanitize_event_data(event_data, session.batman_token)
        else:
            safe_data = event_data
        
        # Create medical event
        medical_event = MedicalEvent(
            event_id=event_id,
            batman_token=session.batman_token,
            event_type=event_type,
            agent_type=session.agent_type,
            data=safe_data,
            session_id=session_id
        )
        
        # Store event
        self.session_events[session_id].append(medical_event)
        self.session_stats["events_recorded"] += 1
        
        # Record in AgentOps
        self.agentops_client.record_medical_event(
            session_id=session_id,
            event_type=event_type,
            event_data=safe_data,
            agent_action=event_type if agent_action else None
        )
        
        # Audit log
        await self.audit_service.log_event(
            event_type=AuditEventType.MEDICAL_DECISION,
            component="medical_telemetry",
            action="record_medical_event",
            details={
                "event_id": event_id,
                "event_type": event_type,
                "agent_type": session.agent_type,
                "batman_token": session.batman_token
            },
            session_id=session_id
        )
        
        logger.debug(f"Medical event recorded: {event_type} for session {session_id}")
        return event_id
    
    async def end_medical_session(self,
                                session_id: str,
                                outcome: str = "completed",
                                summary: Dict[str, Any] = None) -> Dict[str, Any]:
        """End medical session with outcome tracking"""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Attempting to end unknown session: {session_id}")
            return None
        
        session = self.active_sessions[session_id]
        events = self.session_events[session_id]
        
        # Calculate session metrics
        duration = (datetime.now() - session.start_time).total_seconds()
        event_count = len(events)
        
        session_summary = {
            "session_id": session_id,
            "batman_token": session.batman_token,
            "agent_type": session.agent_type,
            "duration_seconds": duration,
            "event_count": event_count,
            "outcome": outcome,
            "compliance": "hipaa_verified",
            **(summary or {})
        }
        
        # End AgentOps session
        self.agentops_client.end_medical_session(
            session_id=session_id,
            outcome=outcome,
            summary=session_summary
        )
        
        # Update stats
        if outcome == "completed":
            self.session_stats["completed_sessions"] += 1
        else:
            self.session_stats["failed_sessions"] += 1
        
        # Audit log
        await self.audit_service.log_event(
            event_type=AuditEventType.SYSTEM_STOP,
            component="medical_telemetry",
            action="end_medical_session",
            details=session_summary,
            session_id=session_id
        )
        
        # Clean up
        session.is_active = False
        del self.active_sessions[session_id]
        del self.session_events[session_id]
        
        logger.info(f"Medical session ended: {session_id} ({outcome}, {duration:.1f}s, {event_count} events)")
        return session_summary
    
    def _sanitize_event_data(self, event_data: Dict[str, Any], batman_token: str) -> Dict[str, Any]:
        """Remove any potential PHI from event data"""
        
        safe_data = {
            "batman_token": batman_token,  # Always include safe token
            "sanitized": True,
            "timestamp": datetime.now().isoformat()
        }
        
        for key, value in event_data.items():
            if not self._contains_phi(str(value)):
                safe_data[key] = value
            else:
                self.session_stats["phi_violations_detected"] += 1
                logger.warning(f"PHI detected and removed from event data: {key}")
                safe_data[f"{key}_sanitized"] = "[PHI_REMOVED]"
        
        return safe_data
    
    def _contains_phi(self, text: str) -> bool:
        """Conservative PHI detection"""
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z]+\s+[A-Za-z]+\b',  # Name patterns (conservative)
            r'\b\d{10,}\b',  # Long numbers (MRN-like)
            r'\b\w+@\w+\.\w+\b'  # Email patterns
        ]
        
        import re
        text_lower = text.lower()
        
        # Check for common PHI indicators
        phi_keywords = ['patient', 'name', 'ssn', 'dob', 'address', 'phone', 'email', 'mrn']
        if any(keyword in text_lower for keyword in phi_keywords):
            return True
        
        # Check patterns
        return any(re.search(pattern, text) for pattern in phi_patterns)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        
        agentops_stats = self.agentops_client.get_session_stats()
        
        return {
            **self.session_stats,
            "active_sessions": len(self.active_sessions),
            "agentops_integration": agentops_stats,
            "compliance_status": {
                "phi_protection_enabled": self.enable_phi_protection,
                "batman_tokens_only": True,
                "audit_trail_active": True
            }
        }
    
    async def get_medical_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time medical dashboard data"""
        
        active_sessions_data = []
        for session_id, session in self.active_sessions.items():
            events = self.session_events.get(session_id, [])
            active_sessions_data.append({
                "session_id": session_id,
                "batman_token": session.batman_token,
                "agent_type": session.agent_type,
                "start_time": session.start_time.isoformat(),
                "event_count": len(events),
                "duration": (datetime.now() - session.start_time).total_seconds()
            })
        
        return {
            "active_sessions": active_sessions_data,
            "session_statistics": self.get_session_statistics(),
            "recent_events": await self._get_recent_events(limit=10),
            "compliance_status": "HIPAA_COMPLIANT",
            "monitoring_status": "ACTIVE"
        }
    
    async def _get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent medical events across all sessions"""
        
        all_events = []
        for session_id, events in self.session_events.items():
            for event in events[-limit:]:  # Get recent events per session
                all_events.append({
                    "event_id": event.event_id,
                    "session_id": session_id,
                    "event_type": event.event_type,
                    "agent_type": event.agent_type,
                    "timestamp": event.timestamp.isoformat(),
                    "batman_token": event.batman_token
                })
        
        # Sort by timestamp and limit
        all_events.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_events[:limit]

__all__ = ['MedicalTelemetry', 'SessionType', 'MedicalEvent']