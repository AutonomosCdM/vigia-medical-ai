"""
Slack Block Templates - Medical Grade Components for VIGIA v1.0
================================================================

Advanced Slack Block Kit templates for medical communication with HIPAA compliance
and Batman tokenization integration.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

# Medical severity constants
class MedicalSeverity(Enum):
    """Medical severity levels with visual indicators"""
    NONE = 0
    LOW = 1  
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

class LPPGrade(Enum):
    """LPP (Lesión por Presión) grading according to NPUAP/EPUAP 2019"""
    GRADE_0 = 0  # No visible injury
    GRADE_1 = 1  # Non-blanchable erythema
    GRADE_2 = 2  # Partial thickness skin loss
    GRADE_3 = 3  # Full thickness skin loss
    GRADE_4 = 4  # Full thickness tissue loss

class VoiceAnalysisIndicator(Enum):
    """Voice analysis emotional indicators"""
    PAIN_LOW = "pain_low"
    PAIN_MODERATE = "pain_moderate" 
    PAIN_HIGH = "pain_high"
    STRESS_LOW = "stress_low"
    STRESS_MODERATE = "stress_moderate"
    STRESS_HIGH = "stress_high"
    DISTRESS = "distress"
    NORMAL = "normal"

@dataclass
class MedicalBlockContext:
    """Context for medical block generation"""
    batman_token: str
    session_id: str
    timestamp: datetime
    severity: MedicalSeverity
    requires_human_review: bool = False
    urgency_level: str = "medium"
    medical_context: Optional[Dict[str, Any]] = None

class VigiaMessageTemplates:
    """
    VIGIA Medical AI message templates with Batman tokenization and medical compliance.
    Provides professional medical-grade Slack components.
    """
    
    # Medical severity color mapping
    SEVERITY_COLORS = {
        MedicalSeverity.NONE: "#36a64f",      # Green
        MedicalSeverity.LOW: "#ffeb3b",       # Yellow  
        MedicalSeverity.MODERATE: "#ff9800",  # Orange
        MedicalSeverity.HIGH: "#f44336",      # Red
        MedicalSeverity.CRITICAL: "#9c27b0"   # Purple
    }
    
    # LPP Grade visualization
    LPP_GRADE_INDICATORS = {
        LPPGrade.GRADE_0: "⚪ Grade 0",
        LPPGrade.GRADE_1: "🟡 Grade 1", 
        LPPGrade.GRADE_2: "🟠 Grade 2",
        LPPGrade.GRADE_3: "🔴 Grade 3",
        LPPGrade.GRADE_4: "⚫ Grade 4"
    }
    
    # Voice analysis visual indicators
    VOICE_INDICATORS = {
        VoiceAnalysisIndicator.PAIN_LOW: "🟢 Pain: Low",
        VoiceAnalysisIndicator.PAIN_MODERATE: "🟡 Pain: Moderate",
        VoiceAnalysisIndicator.PAIN_HIGH: "🔴 Pain: High",
        VoiceAnalysisIndicator.STRESS_LOW: "🟢 Stress: Low", 
        VoiceAnalysisIndicator.STRESS_MODERATE: "🟡 Stress: Moderate",
        VoiceAnalysisIndicator.STRESS_HIGH: "🔴 Stress: High",
        VoiceAnalysisIndicator.DISTRESS: "🆘 Emotional Distress",
        VoiceAnalysisIndicator.NORMAL: "✅ Normal Range"
    }

    @staticmethod
    def create_lpp_detection_alert(
        context: MedicalBlockContext,
        lpp_grade: LPPGrade,
        confidence: float,
        clinical_recommendation: str,
        evidence_level: str = "A"
    ) -> List[Dict[str, Any]]:
        """
        Create LPP detection alert with medical compliance.
        
        Args:
            context: Medical block context with Batman token
            lpp_grade: Detected LPP grade
            confidence: Detection confidence (0.0-1.0)
            clinical_recommendation: Evidence-based recommendation
            evidence_level: Medical evidence level (A/B/C)
            
        Returns:
            List[Dict]: Slack block kit components
        """
        # Determine urgency and color
        severity = MedicalSeverity(lpp_grade.value) if lpp_grade.value <= 4 else MedicalSeverity.CRITICAL
        color = VigiaMessageTemplates.SEVERITY_COLORS[severity]
        grade_indicator = VigiaMessageTemplates.LPP_GRADE_INDICATORS[lpp_grade]
        
        # Create confidence bar visualization
        confidence_bar = VigiaMessageTemplates._create_confidence_bar(confidence)
        
        # Header block
        header_block = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🩺 VIGIA Medical Alert - {grade_indicator}",
                "emoji": True
            }
        }
        
        # Main content block
        content_block = {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Session:* `{context.batman_token[:12]}...`"
                },
                {
                    "type": "mrkdwn", 
                    "text": f"*Confidence:* {confidence:.1%} {confidence_bar}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Evidence Level:* {evidence_level}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Timestamp:* {context.timestamp.strftime('%H:%M:%S')}"
                }
            ]
        }
        
        # Clinical recommendation block
        recommendation_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Clinical Recommendation:*\\n{clinical_recommendation}"
            }
        }
        
        # Action buttons for medical team
        actions_block = {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🏥 Review Case",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": f"review_{context.session_id}",
                    "action_id": "medical_review"
                },
                {
                    "type": "button", 
                    "text": {
                        "type": "plain_text",
                        "text": "📋 Clinical Notes",
                        "emoji": True
                    },
                    "value": f"notes_{context.session_id}",
                    "action_id": "clinical_notes"
                }
            ]
        }
        
        # Add urgency button for high severity
        if severity.value >= 3:
            actions_block["elements"].append({
                "type": "button",
                "text": {
                    "type": "plain_text", 
                    "text": "🚨 Urgent Response",
                    "emoji": True
                },
                "style": "danger",
                "value": f"urgent_{context.session_id}",
                "action_id": "urgent_response"
            })
        
        # Context block with HIPAA compliance note
        context_block = {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🔒 HIPAA Compliant | Batman Token: `{context.batman_token[:8]}...` | NPUAP/EPUAP 2019 Guidelines"
                }
            ]
        }
        
        return [header_block, content_block, recommendation_block, actions_block, context_block]

    @staticmethod
    def create_voice_analysis_alert(
        context: MedicalBlockContext,
        voice_indicators: Dict[str, float],
        emotional_summary: str,
        pain_score: Optional[float] = None,
        stress_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Create voice analysis alert with emotional indicators.
        
        Args:
            context: Medical block context
            voice_indicators: Voice analysis results
            emotional_summary: Summary of emotional state
            pain_score: Pain level score (0.0-1.0)
            stress_score: Stress level score (0.0-1.0)
            
        Returns:
            List[Dict]: Slack block kit components
        """
        # Determine primary indicator
        primary_indicator = VigiaMessageTemplates._determine_primary_voice_indicator(
            voice_indicators, pain_score, stress_score
        )
        
        # Header
        header_block = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🎤 Voice Analysis - {VigiaMessageTemplates.VOICE_INDICATORS[primary_indicator]}",
                "emoji": True
            }
        }
        
        # Scores visualization
        fields = [
            {
                "type": "mrkdwn",
                "text": f"*Session:* `{context.batman_token[:12]}...`"
            },
            {
                "type": "mrkdwn",
                "text": f"*Timestamp:* {context.timestamp.strftime('%H:%M:%S')}"
            }
        ]
        
        # Add pain score if available
        if pain_score is not None:
            pain_bar = VigiaMessageTemplates._create_score_bar(pain_score, "🔴")
            fields.append({
                "type": "mrkdwn",
                "text": f"*Pain Level:* {pain_score:.1%} {pain_bar}"
            })
        
        # Add stress score if available  
        if stress_score is not None:
            stress_bar = VigiaMessageTemplates._create_score_bar(stress_score, "🟡")
            fields.append({
                "type": "mrkdwn",
                "text": f"*Stress Level:* {stress_score:.1%} {stress_bar}"
            })
        
        content_block = {
            "type": "section",
            "fields": fields
        }
        
        # Emotional summary
        summary_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Emotional Analysis:*\\n{emotional_summary}"
            }
        }
        
        # Voice indicators details
        indicators_text = "\\n".join([
            f"• {indicator}: {score:.1%}" 
            for indicator, score in voice_indicators.items()
        ])
        
        indicators_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Voice Indicators:*\\n{indicators_text}"
            }
        }
        
        # Actions for voice analysis
        actions_block = {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🎧 Listen Audio",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": f"audio_{context.session_id}",
                    "action_id": "play_audio"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text", 
                        "text": "📊 Detailed Analysis",
                        "emoji": True
                    },
                    "value": f"voice_details_{context.session_id}",
                    "action_id": "voice_details"
                }
            ]
        }
        
        # Add urgent button for high pain/stress
        if (pain_score and pain_score > 0.7) or (stress_score and stress_score > 0.7):
            actions_block["elements"].append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Pain/Stress Alert", 
                    "emoji": True
                },
                "style": "danger",
                "value": f"pain_alert_{context.session_id}",
                "action_id": "pain_stress_alert"
            })
        
        # Context block
        context_block = {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🔒 HIPAA Compliant | Voice Analysis via Hume AI | Session: `{context.batman_token[:8]}...`"
                }
            ]
        }
        
        return [header_block, content_block, summary_block, indicators_block, actions_block, context_block]

    @staticmethod
    def create_medical_team_coordination(
        context: MedicalBlockContext,
        coordination_type: str,
        team_members: List[str],
        priority: str,
        message: str,
        action_required: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Create medical team coordination message.
        
        Args:
            context: Medical block context
            coordination_type: Type of coordination needed
            team_members: List of team members to notify
            priority: Priority level
            message: Coordination message
            action_required: Whether action is required
            
        Returns:
            List[Dict]: Slack block kit components
        """
        # Priority emoji mapping
        priority_emojis = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🔴",
            "critical": "🆘"
        }
        
        priority_emoji = priority_emojis.get(priority.lower(), "🟡")
        
        # Header
        header_block = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"👥 Medical Team Coordination - {priority_emoji} {priority.upper()}",
                "emoji": True
            }
        }
        
        # Main content
        content_block = {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Type:* {coordination_type}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Priority:* {priority_emoji} {priority.upper()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Session:* `{context.batman_token[:12]}...`"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Time:* {context.timestamp.strftime('%H:%M:%S')}"
                }
            ]
        }
        
        # Message content
        message_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Message:*\\n{message}"
            }
        }
        
        # Team members
        if team_members:
            team_text = " ".join([f"<@{member}>" for member in team_members])
            team_block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Team Members:* {team_text}"
                }
            }
        else:
            team_block = None
        
        # Actions if required
        actions_block = None
        if action_required:
            actions_block = {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "✅ Acknowledge",
                            "emoji": True
                        },
                        "style": "primary",
                        "value": f"ack_{context.session_id}",
                        "action_id": "acknowledge"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "💬 Respond",
                            "emoji": True
                        },
                        "value": f"respond_{context.session_id}",
                        "action_id": "team_respond"
                    }
                ]
            }
            
            # Add urgent escalation for high priority
            if priority.lower() in ["high", "critical"]:
                actions_block["elements"].append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📞 Escalate",
                        "emoji": True
                    },
                    "style": "danger",
                    "value": f"escalate_{context.session_id}",
                    "action_id": "escalate"
                })
        
        # Context
        context_block = {
            "type": "context", 
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🔒 HIPAA Compliant | VIGIA Medical AI | Session: `{context.batman_token[:8]}...`"
                }
            ]
        }
        
        # Build blocks list
        blocks = [header_block, content_block, message_block]
        if team_block:
            blocks.append(team_block)
        if actions_block:
            blocks.append(actions_block)
        blocks.append(context_block)
        
        return blocks

    @staticmethod
    def _create_confidence_bar(confidence: float, length: int = 10) -> str:
        """Create visual confidence bar"""
        filled = int(confidence * length)
        bar = "█" * filled + "░" * (length - filled)
        return f"`{bar}`"
    
    @staticmethod
    def _create_score_bar(score: float, emoji: str = "█", length: int = 8) -> str:
        """Create visual score bar with emoji"""
        filled = int(score * length)
        bar = emoji * filled + "░" * (length - filled)
        return f"`{bar}`"
    
    @staticmethod
    def _determine_primary_voice_indicator(
        voice_indicators: Dict[str, float],
        pain_score: Optional[float] = None,
        stress_score: Optional[float] = None
    ) -> VoiceAnalysisIndicator:
        """Determine primary voice analysis indicator"""
        
        # Check pain level first
        if pain_score is not None:
            if pain_score > 0.7:
                return VoiceAnalysisIndicator.PAIN_HIGH
            elif pain_score > 0.4:
                return VoiceAnalysisIndicator.PAIN_MODERATE  
            elif pain_score > 0.1:
                return VoiceAnalysisIndicator.PAIN_LOW
        
        # Check stress level
        if stress_score is not None:
            if stress_score > 0.7:
                return VoiceAnalysisIndicator.STRESS_HIGH
            elif stress_score > 0.4:
                return VoiceAnalysisIndicator.STRESS_MODERATE
            elif stress_score > 0.1:
                return VoiceAnalysisIndicator.STRESS_LOW
        
        # Check for distress indicators
        distress_indicators = ["sadness", "fear", "anger", "distress"]
        for indicator in distress_indicators:
            if indicator in voice_indicators and voice_indicators[indicator] > 0.6:
                return VoiceAnalysisIndicator.DISTRESS
        
        return VoiceAnalysisIndicator.NORMAL


class SlackBlockBuilder:
    """
    Utility class for building reusable Slack Block Kit components
    with medical-grade styling and HIPAA compliance.
    """
    
    @staticmethod
    def create_medical_header(title: str, emoji: str = "🩺") -> Dict[str, Any]:
        """Create standardized medical header block"""
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True
            }
        }
    
    @staticmethod
    def create_batman_context(batman_token: str, additional_info: str = "") -> Dict[str, Any]:
        """Create HIPAA-compliant context block with Batman token"""
        context_text = f"🔒 HIPAA Compliant | Batman Token: `{batman_token[:8]}...`"
        if additional_info:
            context_text += f" | {additional_info}"
            
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn", 
                    "text": context_text
                }
            ]
        }
    
    @staticmethod
    def create_medical_actions(
        session_id: str,
        primary_action: str = "Review Case",
        include_urgent: bool = False
    ) -> Dict[str, Any]:
        """Create standardized medical action buttons"""
        
        elements = [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"🏥 {primary_action}",
                    "emoji": True
                },
                "style": "primary",
                "value": f"action_{session_id}",
                "action_id": "medical_action"
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "📋 Clinical Notes",
                    "emoji": True  
                },
                "value": f"notes_{session_id}",
                "action_id": "clinical_notes"
            }
        ]
        
        if include_urgent:
            elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Urgent Response",
                    "emoji": True
                },
                "style": "danger",
                "value": f"urgent_{session_id}",
                "action_id": "urgent_response"
            })
        
        return {
            "type": "actions",
            "elements": elements
        }
    
    @staticmethod
    def create_severity_field(severity: MedicalSeverity, label: str = "Severity") -> Dict[str, Any]:
        """Create severity field with visual indicator"""
        color_emojis = {
            MedicalSeverity.NONE: "⚪",
            MedicalSeverity.LOW: "🟡",
            MedicalSeverity.MODERATE: "🟠", 
            MedicalSeverity.HIGH: "🔴",
            MedicalSeverity.CRITICAL: "⚫"
        }
        
        emoji = color_emojis[severity]
        return {
            "type": "mrkdwn",
            "text": f"*{label}:* {emoji} {severity.name.title()}"
        }


# Export main classes
__all__ = [
    'VigiaMessageTemplates',
    'SlackBlockBuilder', 
    'MedicalBlockContext',
    'MedicalSeverity',
    'LPPGrade',
    'VoiceAnalysisIndicator'
]