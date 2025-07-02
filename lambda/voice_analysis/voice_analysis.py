"""
Voice Analysis Agent - Lambda Function
=====================================

AWS Lambda implementation of Voice Analysis Agent using Hume AI for medical voice analysis
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

class VoiceAnalysisAgent:
    """Voice analysis agent for medical emotion and pain assessment"""
    
    def __init__(self):
        self.agent_id = f"voice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def analyze_medical_voice_async(self, voice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voice data for medical assessment"""
        
        batman_token = voice_data.get('batman_token')
        case_id = voice_data.get('case_id')
        
        logger.info(f"🎤 Voice Analysis Agent processing case: {case_id}")
        
        try:
            # Simulate Hume AI voice analysis
            analysis_result = {
                'agent_id': self.agent_id,
                'batman_token': batman_token,
                'case_id': case_id,
                'status': 'completed',
                'voice_analysis': {
                    'pain_indicators': {
                        'vocal_strain': 0.7,
                        'breathing_irregular': 0.6,
                        'speech_hesitation': 0.5
                    },
                    'emotional_state': {
                        'anxiety': 0.6,
                        'distress': 0.7,
                        'fatigue': 0.8
                    },
                    'pain_score_estimated': 6,
                    'requires_follow_up': True
                },
                'completed_at': datetime.now().isoformat()
            }
            
            await self._update_agent_state(analysis_result)
            return analysis_result
            
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
            'agent_type': 'voice_analysis',
            'agent_id': self.agent_id,
            'state_data': json.dumps(state),
            'case_id': state.get('case_id'),
            'status': state.get('status'),
            'updated_at': datetime.now().isoformat()
        }
        
        agents_table.put_item(Item=item)

# Global agent instance
voice_agent = VoiceAnalysisAgent()

def handler(event, context):
    """AWS Lambda handler for Voice Analysis Agent"""
    
    logger.info(f"🎤 Voice Analysis Agent Lambda triggered")
    
    try:
        voice_data = event.get('voice_data', event)
        
        if 'case_id' not in voice_data:
            voice_data['case_id'] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'batman_token' not in voice_data:
            voice_data['batman_token'] = f"BATMAN-VOICE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                voice_agent.analyze_medical_voice_async(voice_data)
            )
        finally:
            loop.close()
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"❌ Voice Analysis Agent error: {str(e)}")
        return {
            'statusCode': 500,
            'body': {'error': str(e)}
        }