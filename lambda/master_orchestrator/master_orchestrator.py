"""
Master Medical Orchestrator - Lambda Function
============================================

AWS Lambda implementation of the Master Medical Orchestrator for the VIGIA Medical AI system.
Coordinates all 9 medical agents for pressure injury detection and analysis.
"""

import json
import logging
import boto3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

# AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
AGENTS_STATE_TABLE = os.environ['AGENTS_STATE_TABLE']
MEDICAL_AUDIT_TABLE = os.environ['MEDICAL_AUDIT_TABLE'] 
LPP_RESULTS_TABLE = os.environ['LPP_RESULTS_TABLE']
MEDICAL_STORAGE_BUCKET = os.environ['MEDICAL_STORAGE_BUCKET']

# DynamoDB tables
agents_table = dynamodb.Table(AGENTS_STATE_TABLE)
audit_table = dynamodb.Table(MEDICAL_AUDIT_TABLE)
results_table = dynamodb.Table(LPP_RESULTS_TABLE)

class MasterMedicalOrchestrator:
    """Master orchestrator for coordinating medical agent workflow"""
    
    def __init__(self):
        self.orchestrator_id = f"master_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_timeout = timedelta(minutes=15)
        
    async def process_medical_case_async(self, medical_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete medical case through the 9-agent workflow
        
        Args:
            medical_case: Medical case data with batman_token, image_path, patient_context
            
        Returns:
            Orchestration result with agent coordination status
        """
        batman_token = medical_case.get('batman_token')
        case_id = medical_case.get('case_id', f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        logger.info(f"🏥 Master Orchestrator processing case: {case_id} with token: {batman_token}")
        
        try:
            # Initialize orchestration state
            orchestration_state = {
                'case_id': case_id,
                'batman_token': batman_token,
                'orchestrator_id': self.orchestrator_id,
                'status': 'processing',
                'started_at': datetime.now().isoformat(),
                'agents_status': {},
                'workflow_stage': 'initialization'
            }
            
            # Store initial state
            await self._update_orchestration_state(orchestration_state)
            
            # Create audit trail entry
            audit_entry = {
                'audit_id': f"audit_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'audit_timestamp': int(datetime.now().timestamp() * 1000),
                'case_id': case_id,
                'batman_token': batman_token,
                'action': 'case_initiated',
                'orchestrator_id': self.orchestrator_id,
                'compliance_status': 'HIPAA_compliant'
            }
            
            audit_table.put_item(Item=audit_entry)
            
            # Validate medical case input
            validation_result = await self._validate_medical_case(medical_case)
            if not validation_result['valid']:
                orchestration_state['status'] = 'validation_failed'
                orchestration_state['error'] = validation_result['error']
                await self._update_orchestration_state(orchestration_state)
                return orchestration_state
            
            # Determine workflow path based on case type
            workflow_path = await self._determine_workflow_path(medical_case)
            orchestration_state['workflow_path'] = workflow_path
            orchestration_state['workflow_stage'] = 'agent_coordination'
            
            # Prepare agent coordination instructions
            agent_instructions = await self._prepare_agent_instructions(medical_case, workflow_path)
            
            # Update state before agent coordination
            await self._update_orchestration_state(orchestration_state)
            
            # Coordinate with specialized agents
            coordination_result = {
                'status': 'completed',
                'workflow_path': workflow_path,
                'agent_instructions': agent_instructions,
                'coordination_timestamp': datetime.now().isoformat(),
                'next_agents': self._get_next_agents(workflow_path),
                'escalation_required': await self._check_escalation_criteria(medical_case)
            }
            
            # Final orchestration state
            orchestration_state['status'] = 'coordination_completed'
            orchestration_state['workflow_stage'] = 'agent_dispatch'
            orchestration_state['coordination_result'] = coordination_result
            orchestration_state['completed_at'] = datetime.now().isoformat()
            
            await self._update_orchestration_state(orchestration_state)
            
            logger.info(f"✅ Master Orchestrator completed case: {case_id}")
            return orchestration_state
            
        except Exception as e:
            error_state = {
                'case_id': case_id,
                'batman_token': batman_token,
                'status': 'error',
                'error': str(e),
                'error_timestamp': datetime.now().isoformat()
            }
            
            await self._update_orchestration_state(error_state)
            logger.error(f"❌ Master Orchestrator error for case {case_id}: {str(e)}")
            return error_state
    
    async def _validate_medical_case(self, medical_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical case data and ensure HIPAA compliance"""
        required_fields = ['batman_token', 'case_id']
        
        for field in required_fields:
            if field not in medical_case:
                return {
                    'valid': False,
                    'error': f'Missing required field: {field}'
                }
        
        # Validate batman token format (HIPAA compliance)
        batman_token = medical_case['batman_token']
        if not batman_token.startswith('BATMAN-'):
            return {
                'valid': False,
                'error': 'Invalid batman token format - HIPAA compliance violation'
            }
        
        return {'valid': True}
    
    async def _determine_workflow_path(self, medical_case: Dict[str, Any]) -> str:
        """Determine the appropriate workflow path based on case characteristics"""
        
        has_image = 'image_path' in medical_case or 'image_data' in medical_case
        has_voice = 'voice_data' in medical_case or 'audio_path' in medical_case
        emergency_indicators = medical_case.get('emergency_indicators', [])
        
        if emergency_indicators:
            return 'emergency_workflow'
        elif has_image and has_voice:
            return 'multimodal_workflow'
        elif has_image:
            return 'image_primary_workflow'
        elif has_voice:
            return 'voice_primary_workflow'
        else:
            return 'assessment_only_workflow'
    
    async def _prepare_agent_instructions(self, medical_case: Dict[str, Any], workflow_path: str) -> Dict[str, Any]:
        """Prepare specific instructions for each agent based on workflow path"""
        
        base_instructions = {
            'case_id': medical_case['case_id'],
            'batman_token': medical_case['batman_token'],
            'workflow_path': workflow_path,
            'orchestrator_id': self.orchestrator_id,
            'priority': self._determine_case_priority(medical_case)
        }
        
        # Workflow-specific instructions
        workflow_instructions = {
            'emergency_workflow': {
                'timeout': 300,  # 5 minutes for emergency
                'priority': 'critical',
                'escalation': 'immediate',
                'agents_sequence': ['image_analysis', 'clinical_assessment', 'communication']
            },
            'multimodal_workflow': {
                'timeout': 900,  # 15 minutes for multimodal
                'priority': 'high',
                'agents_sequence': ['image_analysis', 'voice_analysis', 'clinical_assessment', 'risk_assessment', 'diagnostic', 'protocol', 'communication']
            },
            'image_primary_workflow': {
                'timeout': 600,  # 10 minutes for image-based
                'priority': 'normal',
                'agents_sequence': ['image_analysis', 'monai_review', 'clinical_assessment', 'diagnostic', 'protocol', 'communication']
            },
            'voice_primary_workflow': {
                'timeout': 480,  # 8 minutes for voice-based
                'priority': 'normal',
                'agents_sequence': ['voice_analysis', 'clinical_assessment', 'risk_assessment', 'communication']
            },
            'assessment_only_workflow': {
                'timeout': 360,  # 6 minutes for assessment only
                'priority': 'low',
                'agents_sequence': ['clinical_assessment', 'risk_assessment', 'protocol', 'communication']
            }
        }
        
        return {**base_instructions, **workflow_instructions.get(workflow_path, {})}
    
    def _get_next_agents(self, workflow_path: str) -> List[str]:
        """Get the list of agents to be executed next based on workflow path"""
        
        agent_sequences = {
            'emergency_workflow': ['image_analysis', 'clinical_assessment'],
            'multimodal_workflow': ['image_analysis', 'voice_analysis'],
            'image_primary_workflow': ['image_analysis', 'monai_review'],
            'voice_primary_workflow': ['voice_analysis'],
            'assessment_only_workflow': ['clinical_assessment']
        }
        
        return agent_sequences.get(workflow_path, ['clinical_assessment'])
    
    async def _check_escalation_criteria(self, medical_case: Dict[str, Any]) -> bool:
        """Check if case meets criteria for human escalation"""
        
        escalation_indicators = [
            medical_case.get('confidence_score', 1.0) < 0.6,
            'emergency' in medical_case.get('case_type', '').lower(),
            'grade_4' in str(medical_case.get('initial_assessment', '')).lower(),
            medical_case.get('patient_risk_level') == 'critical'
        ]
        
        return any(escalation_indicators)
    
    def _determine_case_priority(self, medical_case: Dict[str, Any]) -> str:
        """Determine case priority based on medical indicators"""
        
        emergency_keywords = ['emergency', 'urgent', 'critical', 'grade_4', 'infection']
        case_text = str(medical_case).lower()
        
        if any(keyword in case_text for keyword in emergency_keywords):
            return 'critical'
        elif medical_case.get('patient_age', 0) > 75:
            return 'high'
        else:
            return 'normal'
    
    async def _update_orchestration_state(self, state: Dict[str, Any]) -> None:
        """Update orchestration state in DynamoDB"""
        
        item = {
            'batman_token': state['batman_token'],
            'agent_timestamp': int(datetime.now().timestamp() * 1000),
            'agent_type': 'master_orchestrator',
            'agent_id': self.orchestrator_id,
            'state_data': json.dumps(state),
            'case_id': state.get('case_id'),
            'status': state.get('status'),
            'updated_at': datetime.now().isoformat()
        }
        
        agents_table.put_item(Item=item)

# Global orchestrator instance
orchestrator = MasterMedicalOrchestrator()

def handler(event, context):
    """
    AWS Lambda handler for Master Medical Orchestrator
    
    Args:
        event: Lambda event containing medical_case data
        context: Lambda context
        
    Returns:
        Orchestration result
    """
    logger.info(f"🏥 Master Medical Orchestrator Lambda triggered")
    logger.info(f"Event: {json.dumps(event, default=str)}")
    
    try:
        # Extract medical case from event
        if 'medical_case' in event:
            medical_case = event['medical_case']
        else:
            medical_case = event
        
        # Ensure case_id exists
        if 'case_id' not in medical_case:
            medical_case['case_id'] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure batman_token exists (create demo token if missing)
        if 'batman_token' not in medical_case:
            medical_case['batman_token'] = f"BATMAN-DEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Process medical case synchronously (Lambda doesn't support async handlers directly)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                orchestrator.process_medical_case_async(medical_case)
            )
        finally:
            loop.close()
        
        logger.info(f"✅ Orchestration completed successfully")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"❌ Master Orchestrator Lambda error: {str(e)}")
        
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'orchestrator_id': orchestrator.orchestrator_id,
                'timestamp': datetime.now().isoformat()
            }
        }