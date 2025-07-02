"""
diagnostic Agent - Lambda Function
"""

import json
import logging
import boto3
import os
from datetime import datetime
from typing import Dict, Any

dynamodb = boto3.resource('dynamodb')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

AGENTS_STATE_TABLE = os.environ['AGENTS_STATE_TABLE']
agents_table = dynamodb.Table(AGENTS_STATE_TABLE)

class diagnosticAgent:
    def __init__(self):
        self.agent_id = f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        batman_token = data.get('batman_token')
        case_id = data.get('case_id')
        
        result = {
            'agent_id': self.agent_id,
            'batman_token': batman_token,
            'case_id': case_id,
            'status': 'completed',
            'agent_type': 'diagnostic',
            'completed_at': datetime.now().isoformat()
        }
        
        await self._update_agent_state(result)
        return result
    
    async def _update_agent_state(self, state: Dict[str, Any]) -> None:
        item = {
            'batman_token': state['batman_token'],
            'agent_timestamp': int(datetime.now().timestamp() * 1000),
            'agent_type': 'diagnostic',
            'agent_id': self.agent_id,
            'state_data': json.dumps(state),
            'case_id': state.get('case_id'),
            'status': state.get('status'),
            'updated_at': datetime.now().isoformat()
        }
        agents_table.put_item(Item=item)

agent = diagnosticAgent()

def handler(event, context):
    try:
        data = event.get('diagnostic_data', event)
        
        if 'case_id' not in data:
            data['case_id'] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'batman_token' not in data:
            data['batman_token'] = f"BATMAN-TOKEN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(agent.process_async(data))
        finally:
            loop.close()
        
        return {'statusCode': 200, 'body': result}
        
    except Exception as e:
        return {'statusCode': 500, 'body': {'error': str(e)}}
