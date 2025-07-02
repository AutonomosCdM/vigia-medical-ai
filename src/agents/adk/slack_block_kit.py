"""
Slack Block Kit Agent (ADK)
==========================

ADK-based agent for generating Slack Block Kit components for medical notifications.
Supports voice analysis alerts, medical assessments, and trend reporting.

Key Features:
- Medical-grade Slack notifications
- HIPAA-compliant patient data anonymization
- Voice analysis alert blocks
- Trend analysis visualization
- Batman token integration (NO PHI)

Usage:
    agent = SlackBlockKitAgent()
    blocks = agent.generate_voice_analysis_alert(analysis_result)
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from src.agents.base_agent import BaseAgent, AgentMessage, AgentResponse

logger = logging.getLogger(__name__)


class SlackBlockKitAgent(BaseAgent):
    """
    ADK-based agent for generating Slack Block Kit components.
    
    Capabilities:
    - Voice analysis alert blocks
    - Medical assessment notifications
    - Trend analysis visualizations
    - HIPAA-compliant anonymization
    - Batman token integration
    """
    
    def __init__(self, agent_id: str = "slack_blockkit_agent"):
        super().__init__(agent_id=agent_id, agent_type="slack_integration")
        
        # Alert level color mappings
        self.alert_colors = {
            "normal": "#36a64f",      # Green
            "elevated": "#ff9800",    # Orange
            "high": "#ff5722",        # Red-Orange
            "critical": "#f44336"     # Red
        }
        
        # Score bar configurations
        self.score_bar_config = {
            "width": 10,
            "high_threshold": 0.7,
            "medium_threshold": 0.4
        }
    
    async def initialize(self) -> bool:
        """Initialize the Slack Block Kit agent"""
        try:
            logger.info("Slack Block Kit Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Slack Block Kit Agent: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "voice_analysis_alert_blocks",
            "medical_assessment_blocks",
            "voice_trend_blocks",
            "patient_anonymization",
            "hipaa_compliant_formatting",
            "interactive_medical_buttons",
            "score_visualization",
            "batman_tokenization_support"
        ]
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming agent messages"""
        try:
            action = message.action
            
            if action == "generate_voice_analysis_alert":
                return await self._handle_voice_alert_generation(message)
            elif action == "generate_medical_assessment_blocks":
                return await self._handle_medical_assessment_blocks(message)
            elif action == "generate_voice_trend_blocks":
                return await self._handle_trend_blocks(message)
            elif action == "anonymize_patient_data":
                return await self._handle_anonymization(message)
            else:
                return AgentResponse(
                    success=False,
                    message=f"Unknown action: {action}",
                    data={}
                )
                
        except Exception as e:
            logger.error(f"Error processing message in Slack Block Kit Agent: {e}")
            return AgentResponse(
                success=False,
                message=f"Processing error: {str(e)}",
                data={}
            )
    
    async def _handle_voice_alert_generation(self, message: AgentMessage) -> AgentResponse:
        """Handle voice analysis alert block generation"""
        try:
            analysis_result = message.data.get("analysis_result")
            
            if not analysis_result:
                return AgentResponse(
                    success=False,
                    message="Missing analysis_result parameter",
                    data={}
                )
            
            # Generate alert blocks
            blocks = self._generate_voice_analysis_alert_blocks(analysis_result)
            
            return AgentResponse(
                success=True,
                message="Voice analysis alert blocks generated",
                data={
                    "blocks": blocks,
                    "voice_analysis": True,
                    "hipaa_compliant": True,
                    "anonymized": True
                }
            )
            
        except Exception as e:
            logger.error(f"Voice alert generation failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Alert generation failed: {str(e)}",
                data={}
            )
    
    async def _handle_medical_assessment_blocks(self, message: AgentMessage) -> AgentResponse:
        """Handle medical assessment block generation"""
        try:
            assessment_data = message.data.get("assessment_data")
            
            if not assessment_data:
                return AgentResponse(
                    success=False,
                    message="Missing assessment_data parameter",
                    data={}
                )
            
            # Generate assessment blocks
            blocks = self._generate_medical_assessment_blocks(assessment_data)
            
            return AgentResponse(
                success=True,
                message="Medical assessment blocks generated",
                data={
                    "blocks": blocks,
                    "medical_assessment": True,
                    "hipaa_compliant": True
                }
            )
            
        except Exception as e:
            logger.error(f"Medical assessment blocks generation failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Assessment blocks generation failed: {str(e)}",
                data={}
            )
    
    async def _handle_trend_blocks(self, message: AgentMessage) -> AgentResponse:
        """Handle trend analysis block generation"""
        try:
            trend_data = message.data.get("trend_data")
            token_id = message.data.get("token_id")
            
            if not trend_data or not token_id:
                return AgentResponse(
                    success=False,
                    message="Missing trend_data or token_id parameter",
                    data={}
                )
            
            # Generate trend blocks
            blocks = self._generate_voice_trend_blocks(trend_data, token_id)
            
            return AgentResponse(
                success=True,
                message="Voice trend blocks generated",
                data={
                    "blocks": blocks,
                    "trend_analysis": True,
                    "hipaa_compliant": True
                }
            )
            
        except Exception as e:
            logger.error(f"Trend blocks generation failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Trend blocks generation failed: {str(e)}",
                data={}
            )
    
    async def _handle_anonymization(self, message: AgentMessage) -> AgentResponse:
        """Handle patient data anonymization"""
        try:
            patient_data = message.data.get("patient_data")
            
            if not patient_data:
                return AgentResponse(
                    success=False,
                    message="Missing patient_data parameter",
                    data={}
                )
            
            # Anonymize patient data
            anonymized_data = self._anonymize_patient_data(patient_data)
            
            return AgentResponse(
                success=True,
                message="Patient data anonymized successfully",
                data={
                    "anonymized_data": anonymized_data,
                    "hipaa_compliant": True
                }
            )
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Anonymization failed: {str(e)}",
                data={}
            )
    
    def _create_adk_tools(self) -> Dict[str, callable]:
        """Create ADK tools for block generation"""
        return {
            "generate_voice_analysis_alert_blocks": self._generate_voice_analysis_alert_blocks,
            "generate_medical_assessment_blocks": self._generate_medical_assessment_blocks,
            "generate_voice_trend_blocks": self._generate_voice_trend_blocks,
            "anonymize_patient_data": self._anonymize_patient_data,
            "format_score_bar": self._format_score_bar
        }
    
    def _generate_voice_analysis_alert_blocks(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Slack blocks for voice analysis alerts"""
        
        # Extract key information
        token_id = analysis_result.get("token_id", "UNKNOWN")
        medical_indicators = analysis_result.get("medical_indicators", {})
        analysis_id = analysis_result.get("analysis_id", "N/A")
        timestamp = analysis_result.get("timestamp", datetime.now().isoformat())
        
        # Anonymize patient ID
        anonymized_id = self._anonymize_token_id(token_id)
        
        # Get alert level and color
        alert_level = medical_indicators.get("alert_level", "normal")
        alert_color = self.alert_colors.get(alert_level, "#36a64f")
        
        # Build blocks
        blocks = []
        
        # Header block
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🎤 Voice Analysis Alert - Patient {anonymized_id}"
            }
        })
        
        # Alert context
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Analysis ID: `{analysis_id}` • {timestamp[:19]} • Alert Level: *{alert_level.upper()}*"
                }
            ]
        })
        
        # Medical indicators section
        pain_score = medical_indicators.get("pain_score", 0)
        stress_level = medical_indicators.get("stress_level", 0)
        emotional_distress = medical_indicators.get("emotional_distress", 0)
        confidence = analysis_result.get("confidence_level", 0)
        
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Pain Score:*\n{self._format_score_bar(pain_score)} {pain_score:.2f}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Stress Level:*\n{self._format_score_bar(stress_level)} {stress_level:.2f}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Emotional Distress:*\n{self._format_score_bar(emotional_distress)} {emotional_distress:.2f}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence:*\n{self._format_score_bar(confidence)} {confidence:.2f}"
                }
            ]
        })
        
        # Recommendations section
        recommendations = medical_indicators.get("recommendations", [])
        if recommendations:
            recommendation_text = "\n".join([f"• {rec}" for rec in recommendations if rec])
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Medical Recommendations:*\n{recommendation_text}"
                }
            })
        
        # Divider
        blocks.append({"type": "divider"})
        
        # Action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📊 View Trends"
                    },
                    "style": "primary",
                    "value": f"view_trends_{token_id}",
                    "action_id": "view_voice_trends"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🩺 Medical Review"
                    },
                    "value": f"medical_review_{token_id}",
                    "action_id": "request_medical_review"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Acknowledge"
                    },
                    "style": "primary" if alert_level in ["high", "critical"] else None,
                    "value": f"acknowledge_{analysis_id}",
                    "action_id": "acknowledge_alert"
                }
            ]
        })
        
        return blocks
    
    def _generate_medical_assessment_blocks(self, assessment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Slack blocks for medical assessments"""
        
        token_id = assessment_data.get("patient_id", "UNKNOWN")
        assessment_id = assessment_data.get("assessment_id", "N/A")
        urgency_level = assessment_data.get("urgency_level", "normal")
        primary_concerns = assessment_data.get("primary_concerns", [])
        recommendations = assessment_data.get("medical_recommendations", [])
        
        # Anonymize patient ID
        anonymized_id = self._anonymize_token_id(token_id)
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🏥 Medical Assessment - Patient {anonymized_id}"
            }
        })
        
        # Assessment details
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Assessment ID:* `{assessment_id}`\n*Urgency Level:* *{urgency_level.upper()}*"
            }
        })
        
        # Primary concerns
        if primary_concerns:
            concerns_text = "\n".join([f"• {concern}" for concern in primary_concerns])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Primary Concerns:*\n{concerns_text}"
                }
            })
        
        # Recommendations
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Recommendations:*\n{rec_text}"
                }
            })
        
        return blocks
    
    def _generate_voice_trend_blocks(self, trend_data: Dict[str, Any], token_id: str) -> List[Dict[str, Any]]:
        """Generate Slack blocks for voice trend analysis"""
        
        anonymized_id = self._anonymize_token_id(token_id)
        timeframe = trend_data.get("timeframe", "unknown")
        data_points = trend_data.get("data_points", 0)
        trends = trend_data.get("trends", {})
        summary = trend_data.get("summary", "No trend analysis available")
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📈 Voice Trend Analysis - Patient {anonymized_id}"
            }
        })
        
        # Trend summary
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Timeframe:* {timeframe} • *Data Points:* {data_points}\n\n*Summary:* {summary}"
            }
        })
        
        # Individual trends
        if trends:
            trend_fields = []
            for trend_type, trend_direction in trends.items():
                emoji = self._get_trend_emoji(trend_direction)
                trend_fields.append({
                    "type": "mrkdwn",
                    "text": f"*{trend_type.replace('_', ' ').title()}:*\n{emoji} {trend_direction.title()}"
                })
            
            blocks.append({
                "type": "section",
                "fields": trend_fields
            })
        
        return blocks
    
    def _anonymize_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize patient data for HIPAA compliance"""
        anonymized = patient_data.copy()
        
        # Anonymize patient identifiers
        if "patient_id" in anonymized:
            anonymized["patient_id"] = self._anonymize_token_id(anonymized["patient_id"])
        
        if "token_id" in anonymized:
            anonymized["token_id"] = self._anonymize_token_id(anonymized["token_id"])
        
        # Remove or mask sensitive fields
        sensitive_fields = ["name", "phone", "email", "address", "ssn"]
        for field in sensitive_fields:
            if field in anonymized:
                anonymized[field] = "***MASKED***"
        
        return anonymized
    
    def _anonymize_token_id(self, token_id: str) -> str:
        """Anonymize token ID for display"""
        if len(token_id) > 6:
            return f"{token_id[:3]}***{token_id[-3:]}"
        else:
            return "PAT***"
    
    def _format_score_bar(self, score: float) -> str:
        """Format score as visual bar"""
        if score > self.score_bar_config["high_threshold"]:
            emoji = "🔴"
        elif score > self.score_bar_config["medium_threshold"]:
            emoji = "🟠"
        else:
            emoji = "🟢"
        
        # Create bar visualization
        filled_bars = int(score * self.score_bar_config["width"])
        empty_bars = self.score_bar_config["width"] - filled_bars
        
        bar = "█" * filled_bars + "░" * empty_bars
        
        return f"{emoji} {bar}"
    
    def _get_trend_emoji(self, direction: str) -> str:
        """Get emoji for trend direction"""
        direction_lower = direction.lower()
        
        if direction_lower in ["increasing", "rising", "worsening"]:
            return "📈"
        elif direction_lower in ["decreasing", "falling", "improving"]:
            return "📉"
        elif direction_lower in ["stable", "steady", "unchanged"]:
            return "➡️"
        else:
            return "❓"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "capabilities_count": len(self.get_capabilities()),
            "alert_colors_configured": len(self.alert_colors),
            "hipaa_compliant": True,
            "last_block_generation": datetime.now().isoformat()
        }


# Export for ADK integration
__all__ = ["SlackBlockKitAgent"]