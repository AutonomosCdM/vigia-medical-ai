"""
Communication Agent - Lambda Function
====================================

AWS Lambda implementation of Communication Agent for WhatsApp/Slack coordination
"""

import json
import logging
import boto3
import os
from datetime import datetime
from typing import Dict, Any

# AWS clients
dynamodb = boto3.resource('dynamodb')

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
AGENTS_STATE_TABLE = os.environ['AGENTS_STATE_TABLE']
MEDICAL_AUDIT_TABLE = os.environ['MEDICAL_AUDIT_TABLE']

# DynamoDB tables
agents_table = dynamodb.Table(AGENTS_STATE_TABLE)
audit_table = dynamodb.Table(MEDICAL_AUDIT_TABLE)

class CommunicationAgent:
    """Communication agent for medical team coordination"""
    
    def __init__(self):
        self.agent_id = f"communication_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def send_medical_notifications_async(self, communication_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send medical notifications to appropriate channels"""
        
        batman_token = communication_data.get('batman_token')
        case_id = communication_data.get('case_id')
        
        logger.info(f"=Þ Communication Agent processing case: {case_id}")
        
        try:
            # Simulate medical communication
            communication_result = {
                'agent_id': self.agent_id,
                'batman_token': batman_token,
                'case_id': case_id,
                'status': 'completed',
                'communications_sent': {
                    'slack_notifications': {
                        'medical_team_channel': 'sent',
                        'urgent_alerts_channel': 'sent' if communication_data.get('urgent', False) else 'skipped'
                    },
                    'whatsapp_updates': {
                        'patient_notification': 'sent',
                        'family_update': 'sent'
                    },
                    'audit_trail': {
                        'hipaa_compliant': True,
                        'phi_protected': True,
                        'batman_token_used': batman_token
                    }
                },
                'completed_at': datetime.now().isoformat()
            }
            
            await self._update_agent_state(communication_result)
            return communication_result
            
        except Exception as e:
            error_result = {
                'agent_id': self.agent_id,
                'batman_token': batman_token,
                'case_id': case_id,
                'status': 'error',
                'error': str(e)
            }
            
            await self._update_agent_state(error_result)
            return error_result
    
    async def _update_agent_state(self, state: Dict[str, Any]) -> None:
        """Update agent state in DynamoDB"""
        
        item = {
            'batman_token': state['batman_token'],
            'agent_timestamp': int(datetime.now().timestamp() * 1000),
            'agent_type': 'communication',
            'agent_id': self.agent_id,
            'state_data': json.dumps(state),
            'case_id': state.get('case_id'),
            'status': state.get('status'),
            'updated_at': datetime.now().isoformat()
        }
        
        agents_table.put_item(Item=item)

# Global agent instance
communication_agent = CommunicationAgent()

def handler(event, context):
    """AWS Lambda handler for Communication Agent"""
    
    logger.info(f"=Þ Communication Agent Lambda triggered")
    
    try:
        communication_data = event.get('communication_data', event)
        
        if 'case_id' not in communication_data:
            communication_data['case_id'] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'batman_token' not in communication_data:
            communication_data['batman_token'] = f"BATMAN-COMM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                communication_agent.send_medical_notifications_async(communication_data)
            )
        finally:
            loop.close()
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"L Communication Agent error: {str(e)}")
        return {
            'statusCode': 500,
            'body': {'error': str(e)}
        }