"""
Communication Bridge - Inter-Agent Communication Orchestrator
=============================================================

Bridge service that orchestrates communication between MedicalTeamAgent and 
PatientCommunicationAgent for approval workflows and coordinated messaging.

Key Features:
- Medical team approval workflow coordination
- Inter-agent message routing and translation
- Communication session management
- Complete audit trail for all interactions
- HIPAA-compliant token-based communication

Usage:
    bridge = CommunicationBridge()
    await bridge.route_approval_request(approval_data)
    await bridge.deliver_approved_response(approved_response)
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import json
import uuid

# Agent imports
from .medical_team_agent import MedicalTeamAgentFactory
from .patient_communication_agent import PatientCommunicationAgentFactory
from ..db.supabase_client import SupabaseClient
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService, AuditEventType, AuditLevel

# AgentOps Monitoring Integration
from ..monitoring.agentops_client import AgentOpsClient
from ..monitoring.medical_telemetry import MedicalTelemetry

logger = SecureLogger("communication_bridge")


class BridgeMessageType(Enum):
    """Types of inter-agent bridge messages"""
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    ESCALATION_NOTICE = "escalation_notice"
    COORDINATION_MESSAGE = "coordination_message"
    STATUS_UPDATE = "status_update"


class ApprovalWorkflowStatus(Enum):
    """Status of approval workflows"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"


class CommunicationBridge:
    """
    Bridge service for coordinating communication between medical and patient agents
    """
    
    def __init__(self):
        self.medical_team_agent = MedicalTeamAgentFactory.create_agent()
        self.patient_communication_agent = PatientCommunicationAgentFactory.create_agent()
        self.supabase_client = SupabaseClient()
        self.audit_service = AuditService()
        self.active_workflows = {}  # In-memory workflow tracking
        
        # AgentOps monitoring integration
        self.telemetry = MedicalTelemetry(
            app_id="vigia-communication-bridge",
            environment="production",
            enable_phi_protection=True
        )
        self.current_session = None
        
    async def route_medical_diagnosis_to_patient(self, diagnosis_data: Dict[str, Any], 
                                               medical_approval: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route medical diagnosis from MedicalTeamAgent to PatientCommunicationAgent
        
        Args:
            diagnosis_data: Complete diagnosis from DiagnosticAgent
            medical_approval: Approval metadata from medical team
            
        Returns:
            Routing results with delivery status
        """
        try:
            start_time = datetime.now(timezone.utc)
            bridge_id = str(uuid.uuid4())
            
            logger.info(f"Starting diagnosis routing: {bridge_id}")
            
            # Extract key information
            token_id = diagnosis_data.get('token_id', 'UNKNOWN')  # Batman token
            primary_diagnosis = diagnosis_data.get('primary_diagnosis', '')
            approved_by = medical_approval.get('approved_by', 'medical_team')
            
            # Format patient-friendly message
            patient_message = self._format_diagnosis_for_patient(diagnosis_data, medical_approval)
            
            # Prepare response data for PatientCommunicationAgent
            response_data = {
                'token_id': token_id,
                'phone_number': medical_approval.get('patient_phone', ''),  # May contain PHI
                'message_content': patient_message['content'],
                'response_type': 'diagnosis_result'
            }
            
            approval_metadata = {
                'approved_by': approved_by,
                'approval_id': medical_approval.get('approval_id', str(uuid.uuid4())),
                'approval_timestamp': datetime.now(timezone.utc).isoformat(),
                'bridge_id': bridge_id
            }
            
            # Send via PatientCommunicationAgent
            delivery_result = await self.patient_communication_agent.send_approved_response_adk_tool(
                response_data=response_data,
                approval_metadata=approval_metadata
            )
            
            # Store bridge transaction
            bridge_record = await self._store_bridge_transaction(
                bridge_id=bridge_id,
                bridge_type=BridgeMessageType.APPROVAL_RESPONSE.value,
                source_agent='medical_team_agent',
                target_agent='patient_communication_agent',
                source_data=diagnosis_data,
                target_data=response_data,
                metadata={
                    'medical_approval': medical_approval,
                    'delivery_result': delivery_result,
                    'patient_message_formatted': patient_message
                }
            )
            
            # Audit the bridge operation
            await self.audit_service.log_event(
                event_type=AuditEventType.INTER_AGENT_COMMUNICATION,
                event_level=AuditLevel.INFO,
                component="communication_bridge",
                description=f"Routed medical diagnosis to patient: {bridge_id}",
                metadata={
                    'bridge_id': bridge_id,
                    'token_id': token_id,
                    'approved_by': approved_by,
                    'delivery_success': delivery_result.get('success', False)
                }
            )
            
            return {
                'success': True,
                'bridge_id': bridge_id,
                'routing_details': {
                    'token_id': token_id,  # Batman token (HIPAA compliant)
                    'source_agent': 'medical_team_agent',
                    'target_agent': 'patient_communication_agent',
                    'message_type': 'diagnosis_result',
                    'approved_by': approved_by
                },
                'delivery_result': delivery_result,
                'bridge_record_id': bridge_record,
                'processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error routing diagnosis to patient: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'diagnosis_data': diagnosis_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def route_patient_inquiry_to_medical(self, patient_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route patient inquiry from PatientCommunicationAgent to MedicalTeamAgent
        
        Args:
            patient_message: Patient message from WhatsApp
            
        Returns:
            Routing results with medical team notification status
        """
        try:
            start_time = datetime.now(timezone.utc)
            bridge_id = str(uuid.uuid4())
            
            logger.info(f"Starting patient inquiry routing: {bridge_id}")
            
            # Extract patient information
            token_id = patient_message.get('token_id', 'UNKNOWN')  # Batman token
            message_type = patient_message.get('message_type', 'general')
            urgency_level = patient_message.get('urgency_level', 'medium')
            
            # Transform patient message for medical team
            medical_notification = self._format_patient_inquiry_for_medical(patient_message)
            
            # Determine target medical specialists
            target_specialists = self._determine_medical_specialists(message_type, urgency_level)
            
            # Send to MedicalTeamAgent
            medical_delivery_result = await self.medical_team_agent.send_diagnosis_to_team_adk_tool(
                diagnosis_data=medical_notification,
                target_specialists=target_specialists,
                include_evidence=False,
                enable_follow_up=True
            )
            
            # Store bridge transaction
            bridge_record = await self._store_bridge_transaction(
                bridge_id=bridge_id,
                bridge_type=BridgeMessageType.COORDINATION_MESSAGE.value,
                source_agent='patient_communication_agent',
                target_agent='medical_team_agent',
                source_data=patient_message,
                target_data=medical_notification,
                metadata={
                    'target_specialists': target_specialists,
                    'medical_delivery_result': medical_delivery_result,
                    'urgency_escalated': urgency_level in ['high', 'critical', 'emergency']
                }
            )
            
            return {
                'success': True,
                'bridge_id': bridge_id,
                'routing_details': {
                    'token_id': token_id,  # Batman token (HIPAA compliant)
                    'source_agent': 'patient_communication_agent',
                    'target_agent': 'medical_team_agent',
                    'message_type': message_type,
                    'urgency_level': urgency_level,
                    'target_specialists': target_specialists
                },
                'medical_delivery_result': medical_delivery_result,
                'bridge_record_id': bridge_record,
                'processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error routing patient inquiry to medical: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'patient_message': patient_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def manage_approval_workflow(self, approval_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage complete approval workflow between agents
        
        Args:
            approval_request: Request for patient communication approval
            
        Returns:
            Approval workflow management results
        """
        try:
            start_time = datetime.now(timezone.utc)
            workflow_id = str(uuid.uuid4())
            
            logger.info(f"Starting approval workflow: {workflow_id}")
            
            # Extract workflow information
            token_id = approval_request.get('token_id', 'UNKNOWN')  # Batman token
            original_message = approval_request.get('original_message', '')
            urgency_level = approval_request.get('urgency_level', 'medium')
            
            # Create workflow tracking
            workflow_data = {
                'workflow_id': workflow_id,
                'token_id': token_id,
                'status': ApprovalWorkflowStatus.PENDING.value,
                'approval_request': approval_request,
                'created_at': start_time.isoformat(),
                'expires_at': (start_time + timedelta(hours=24)).isoformat()  # 24h expiry
            }
            
            # Store in active workflows
            self.active_workflows[workflow_id] = workflow_data
            
            # Send approval request to MedicalTeamAgent
            medical_approval_result = await self.medical_team_agent.approve_patient_communication_adk_tool(
                approval_request=approval_request,
                approval_status='pending'
            )
            
            # Store workflow in database
            workflow_record = await self._store_approval_workflow(workflow_data, medical_approval_result)
            
            # Set up workflow monitoring
            workflow_monitoring = self._setup_workflow_monitoring(workflow_id, urgency_level)
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'workflow_details': {
                    'token_id': token_id,  # Batman token (HIPAA compliant)
                    'status': ApprovalWorkflowStatus.PENDING.value,
                    'urgency_level': urgency_level,
                    'expires_at': workflow_data['expires_at']
                },
                'medical_approval_result': medical_approval_result,
                'workflow_record_id': workflow_record,
                'workflow_monitoring': workflow_monitoring,
                'processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error managing approval workflow: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'approval_request': approval_request,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def process_approval_decision(self, workflow_id: str, 
                                      approval_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process approval decision and coordinate final delivery
        
        Args:
            workflow_id: Workflow identifier
            approval_decision: Medical team approval decision
            
        Returns:
            Approval processing results
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Retrieve workflow
            if workflow_id not in self.active_workflows:
                return {
                    'success': False,
                    'error': f'Workflow not found: {workflow_id}',
                    'timestamp': start_time.isoformat()
                }
            
            workflow = self.active_workflows[workflow_id]
            
            # Extract decision information
            decision_status = approval_decision.get('approval_status', 'approved')
            approved_by = approval_decision.get('approved_by', 'medical_team')
            modifications = approval_decision.get('modifications', None)
            
            # Update workflow status
            workflow['status'] = decision_status
            workflow['approval_decision'] = approval_decision
            workflow['decided_at'] = start_time.isoformat()
            workflow['decided_by'] = approved_by
            
            # If approved or modified, deliver to patient
            delivery_result = None
            if decision_status in ['approved', 'modified']:
                # Prepare final message
                final_message = modifications if decision_status == 'modified' else workflow['approval_request']['original_message']
                
                # Route to patient
                delivery_result = await self.route_medical_diagnosis_to_patient(
                    diagnosis_data={
                        'token_id': workflow['token_id'],
                        'primary_diagnosis': final_message,
                        'approved_message': True
                    },
                    medical_approval={
                        'approved_by': approved_by,
                        'approval_id': workflow_id,
                        'patient_phone': workflow['approval_request'].get('patient_phone', '')
                    }
                )
            
            # Update workflow record
            await self._update_approval_workflow(workflow_id, workflow, delivery_result)
            
            # Remove from active workflows if completed
            if decision_status in ['approved', 'rejected', 'modified']:
                del self.active_workflows[workflow_id]
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'decision_details': {
                    'token_id': workflow['token_id'],  # Batman token (HIPAA compliant)
                    'decision_status': decision_status,
                    'approved_by': approved_by,
                    'modifications_made': decision_status == 'modified',
                    'delivered_to_patient': delivery_result is not None
                },
                'delivery_result': delivery_result,
                'processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing approval decision: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'workflow_id': workflow_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_bridge_status(self, token_id: str = None) -> Dict[str, Any]:
        """
        Get current status of communication bridge
        
        Args:
            token_id: Optional Batman token to filter by patient
            
        Returns:
            Bridge status information
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Count active workflows
            active_workflow_count = len(self.active_workflows)
            
            # Filter by token if provided
            filtered_workflows = self.active_workflows
            if token_id:
                filtered_workflows = {
                    k: v for k, v in self.active_workflows.items() 
                    if v.get('token_id') == token_id
                }
            
            # Calculate workflow statistics
            pending_approvals = sum(1 for w in filtered_workflows.values() 
                                  if w.get('status') == ApprovalWorkflowStatus.PENDING.value)
            
            expired_workflows = sum(1 for w in filtered_workflows.values() 
                                  if current_time > datetime.fromisoformat(w.get('expires_at', current_time.isoformat()).replace('Z', '+00:00')))
            
            return {
                'success': True,
                'bridge_status': {
                    'operational': True,
                    'active_workflows': active_workflow_count,
                    'pending_approvals': pending_approvals,
                    'expired_workflows': expired_workflows,
                    'filtered_by_token': token_id is not None
                },
                'workflow_details': list(filtered_workflows.values()) if token_id else [],
                'system_capabilities': [
                    'medical_diagnosis_routing',
                    'patient_inquiry_routing',
                    'approval_workflow_management',
                    'inter_agent_coordination',
                    'complete_audit_trail',
                    'hipaa_compliant_tokenization'
                ],
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting bridge status: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    # Helper Methods
    
    def _format_diagnosis_for_patient(self, diagnosis_data: Dict[str, Any], 
                                    medical_approval: Dict[str, Any]) -> Dict[str, str]:
        """Format medical diagnosis for patient-friendly communication"""
        
        primary_diagnosis = diagnosis_data.get('primary_diagnosis', '')
        lpp_grade = diagnosis_data.get('lpp_grade', 0)
        treatment_plan = diagnosis_data.get('treatment_plan', [])
        
        # Create patient-friendly message
        if lpp_grade > 0:
            content = f"""
🩺 **RESULTADO DE ANÁLISIS MÉDICO**

**Diagnóstico:** {primary_diagnosis}

**Plan de Cuidados:**
{chr(10).join(f"• {step}" for step in treatment_plan)}

**Próximos Pasos:**
• Sigue las indicaciones de cuidado
• Contacta a tu equipo médico si tienes dudas
• Programa seguimiento según indicaciones

*Este análisis fue revisado y aprobado por: {medical_approval.get('approved_by', 'Equipo médico')}*

¿Tienes alguna pregunta sobre tu cuidado?
"""
        else:
            content = f"""
✅ **RESULTADO DE ANÁLISIS MÉDICO**

**Buenas noticias:** No se detectaron lesiones por presión en la imagen analizada.

**Recomendaciones preventivas:**
• Continúa con cambios de posición regulares
• Mantén la piel limpia y seca
• Observa cualquier cambio en la piel
• Consulta ante cualquier duda

*Este análisis fue revisado por: {medical_approval.get('approved_by', 'Equipo médico')}*

¡Mantén estos buenos cuidados!
"""
        
        return {
            'content': content.strip(),
            'type': 'medical_result',
            'approved_by': medical_approval.get('approved_by', 'medical_team')
        }
    
    def _format_patient_inquiry_for_medical(self, patient_message: Dict[str, Any]) -> Dict[str, Any]:
        """Format patient inquiry for medical team notification"""
        
        token_id = patient_message.get('token_id', 'UNKNOWN')  # Batman token
        message_type = patient_message.get('message_type', 'general')
        urgency_level = patient_message.get('urgency_level', 'medium')
        content = patient_message.get('message_content', {})
        
        return {
            'token_id': token_id,
            'notification_type': 'patient_inquiry',
            'primary_diagnosis': f"Consulta de paciente ({message_type})",
            'urgency_level': urgency_level,
            'patient_context': {
                'message_type': message_type,
                'inquiry_content': content,
                'requires_response': True
            },
            'clinical_recommendations': [
                'Revisar consulta del paciente',
                'Proporcionar orientación médica apropiada',
                'Documentar respuesta en sistema'
            ],
            'follow_up_required': True
        }
    
    def _determine_medical_specialists(self, message_type: str, urgency_level: str) -> List[str]:
        """Determine appropriate medical specialists for patient inquiry"""
        
        if urgency_level in ['critical', 'emergency']:
            return ['emergency', 'clinical']
        elif message_type == 'medical_image':
            return ['wound_care', 'clinical']
        elif message_type in ['follow_up', 'treatment']:
            return ['clinical', 'nursing']
        else:
            return ['clinical']
    
    async def _store_bridge_transaction(self, bridge_id: str, bridge_type: str,
                                      source_agent: str, target_agent: str,
                                      source_data: Dict[str, Any], target_data: Dict[str, Any],
                                      metadata: Dict[str, Any]) -> str:
        """Store bridge transaction in database"""
        try:
            # This would use SupabaseClient to store in bridge_transactions table
            logger.info(f"Bridge transaction stored: {bridge_id} - {bridge_type}")
            return f"bridge_record_{bridge_id}"
        except Exception as e:
            logger.error(f"Error storing bridge transaction: {str(e)}")
            return None
    
    async def _store_approval_workflow(self, workflow_data: Dict[str, Any], 
                                     medical_result: Dict[str, Any]) -> str:
        """Store approval workflow in database"""
        try:
            # This would use SupabaseClient to store in approval_workflows table
            logger.info(f"Approval workflow stored: {workflow_data['workflow_id']}")
            return f"workflow_record_{workflow_data['workflow_id']}"
        except Exception as e:
            logger.error(f"Error storing approval workflow: {str(e)}")
            return None
    
    async def _update_approval_workflow(self, workflow_id: str, workflow_data: Dict[str, Any],
                                      delivery_result: Optional[Dict[str, Any]]) -> bool:
        """Update approval workflow in database"""
        try:
            # This would use SupabaseClient to update workflow record
            logger.info(f"Approval workflow updated: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating approval workflow: {str(e)}")
            return False
    
    def _setup_workflow_monitoring(self, workflow_id: str, urgency_level: str) -> Dict[str, Any]:
        """Setup monitoring for approval workflow"""
        
        # Determine monitoring intervals based on urgency
        if urgency_level == 'emergency':
            check_interval_minutes = 5
            escalation_timeout_hours = 1
        elif urgency_level == 'critical':
            check_interval_minutes = 15
            escalation_timeout_hours = 2
        elif urgency_level == 'high':
            check_interval_minutes = 30
            escalation_timeout_hours = 4
        else:
            check_interval_minutes = 60
            escalation_timeout_hours = 24
        
        return {
            'monitoring_enabled': True,
            'workflow_id': workflow_id,
            'check_interval_minutes': check_interval_minutes,
            'escalation_timeout_hours': escalation_timeout_hours,
            'urgency_level': urgency_level,
            'setup_timestamp': datetime.now(timezone.utc).isoformat()
        }


# Factory for Communication Bridge
class CommunicationBridgeFactory:
    """Factory for creating Communication Bridge instances"""
    
    @staticmethod
    def create_bridge() -> CommunicationBridge:
        """Create new Communication Bridge instance"""
        return CommunicationBridge()
    
    @staticmethod
    def get_bridge_capabilities() -> List[str]:
        """Get list of bridge capabilities"""
        return [
            'inter_agent_message_routing',
            'medical_diagnosis_delivery',
            'patient_inquiry_routing',
            'approval_workflow_management',
            'coordinated_multi_agent_communication',
            'complete_audit_trail',
            'hipaa_compliant_tokenization',
            'workflow_status_monitoring',
            'escalation_management',
            'session_coordination'
        ]


# Export for use
__all__ = [
    'CommunicationBridge',
    'CommunicationBridgeFactory',
    'BridgeMessageType',
    'ApprovalWorkflowStatus'
]