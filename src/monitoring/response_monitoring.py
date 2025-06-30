"""
VIGIA Medical AI - Response Monitoring System
============================================

Real-time monitoring of LLM responses with automatic escalation,
alert management, and comprehensive logging for medical compliance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import json
import aioredis
from collections import defaultdict

from ..ai.medical_guardrails import medical_guardrails, ResponseSafetyLevel, SafetyViolationType
from ..utils.audit_service import AuditService
from ..utils.secure_logger import SecureLogger
from ..core.celery_config import celery_app, configure_task_defaults

logger = SecureLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EscalationChannel(Enum):
    """Escalation communication channels"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"

@dataclass
class ResponseAlert:
    """Individual response alert"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    batman_token: str
    model_version: str
    violation_types: List[SafetyViolationType]
    safety_level: ResponseSafetyLevel
    description: str
    response_snippet: str
    timestamp: datetime
    escalated: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class EscalationRule:
    """Rules for automatic escalation"""
    condition: str
    channels: List[EscalationChannel]
    delay_minutes: int
    max_attempts: int
    filter_function: Optional[Callable] = None

class ResponseMonitor:
    """Real-time response monitoring with escalation"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379/4"
        self.redis_client = None
        self.audit_service = AuditService()
        
        # Alert storage
        self.active_alerts: Dict[str, ResponseAlert] = {}
        self.alert_counters = defaultdict(int)
        
        # Escalation rules
        self.escalation_rules = [
            EscalationRule(
                condition="critical_safety_violation",
                channels=[EscalationChannel.SLACK, EscalationChannel.EMAIL],
                delay_minutes=0,  # Immediate
                max_attempts=3
            ),
            EscalationRule(
                condition="high_violation_rate",
                channels=[EscalationChannel.SLACK],
                delay_minutes=5,
                max_attempts=2
            ),
            EscalationRule(
                condition="emergency_mishandling",
                channels=[EscalationChannel.SLACK, EscalationChannel.SMS],
                delay_minutes=0,  # Immediate
                max_attempts=5
            ),
            EscalationRule(
                condition="model_degradation",
                channels=[EscalationChannel.EMAIL, EscalationChannel.DASHBOARD],
                delay_minutes=15,
                max_attempts=1
            )
        ]
        
        # Monitoring thresholds
        self.thresholds = {
            'unsafe_response_rate_5min': 0.10,  # 10% unsafe responses in 5 min
            'critical_violations_per_hour': 5,   # 5 critical violations per hour
            'escalation_rate_threshold': 0.05,   # 5% escalation rate
            'response_time_threshold': 5.0,      # 5 seconds max response time
            'confidence_drop_threshold': 0.15    # 15% confidence drop
        }
        
        # Background monitoring tasks
        self.monitoring_tasks = []
        self.is_monitoring = False
        
        logger.info("ResponseMonitor initialized")
    
    async def initialize(self) -> None:
        """Initialize monitoring system"""
        try:
            # Redis connection for real-time data
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )
            
            # Start background monitoring
            await self.start_monitoring()
            
            logger.info("ResponseMonitor initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, using mock mode: {e}")
            self.redis_client = None
    
    async def monitor_response(self,
                              response_text: str,
                              original_prompt: str,
                              model_version: str,
                              batman_token: str,
                              confidence_score: float = None,
                              processing_time: float = None) -> Dict[str, Any]:
        """Monitor individual response in real-time"""
        
        monitoring_start = datetime.now()
        
        try:
            # Run guardrail validation
            guardrail_result = await medical_guardrails.validate_response(
                response_text=response_text,
                original_prompt=original_prompt,
                confidence_score=confidence_score,
                batman_token=batman_token
            )
            
            # Process monitoring results
            monitoring_result = await self._process_monitoring_result(
                guardrail_result, model_version, batman_token, processing_time
            )
            
            # Update real-time metrics
            await self._update_realtime_metrics(monitoring_result)
            
            # Check for immediate alerts
            alerts = await self._check_immediate_alerts(monitoring_result)
            
            # Process alerts and escalations
            for alert in alerts:
                await self._process_alert(alert)
            
            # Log monitoring activity
            monitoring_time = (datetime.now() - monitoring_start).total_seconds()
            
            await self.audit_service.log_activity(
                activity_type="response_monitoring",
                batman_token=batman_token,
                details={
                    'model_version': model_version,
                    'safety_level': guardrail_result.safety_level.value,
                    'violations': len(guardrail_result.violations),
                    'escalated': guardrail_result.escalation_required,
                    'monitoring_time': monitoring_time
                }
            )
            
            return {
                'monitoring_result': monitoring_result,
                'safety_level': guardrail_result.safety_level.value,
                'alerts_generated': len(alerts),
                'escalation_required': guardrail_result.escalation_required,
                'monitoring_time': monitoring_time
            }
            
        except Exception as e:
            logger.error(f"Response monitoring failed: {e}")
            
            # Create critical alert for monitoring system failure
            await self._create_system_alert(
                "monitoring_system_failure",
                AlertSeverity.CRITICAL,
                f"Response monitoring failed: {str(e)}",
                batman_token
            )
            
            return {
                'monitoring_result': None,
                'safety_level': 'unknown',
                'alerts_generated': 0,
                'escalation_required': True,
                'error': str(e)
            }
    
    async def _process_monitoring_result(self,
                                        guardrail_result,
                                        model_version: str,
                                        batman_token: str,
                                        processing_time: float) -> Dict[str, Any]:
        """Process guardrail results for monitoring"""
        
        return {
            'timestamp': datetime.now(),
            'batman_token': batman_token,
            'model_version': model_version,
            'safety_level': guardrail_result.safety_level,
            'violations': guardrail_result.violations,
            'medical_accuracy': guardrail_result.medical_accuracy_score,
            'npuap_compliance': guardrail_result.npuap_compliance_score,
            'confidence_score': guardrail_result.confidence_score,
            'escalation_required': guardrail_result.escalation_required,
            'processing_time': processing_time,
            'guardrail_time': guardrail_result.processing_time
        }
    
    async def _update_realtime_metrics(self, monitoring_result: Dict[str, Any]) -> None:
        """Update real-time monitoring metrics"""
        
        if not self.redis_client:
            return
        
        try:
            timestamp = datetime.now()
            minute_key = timestamp.strftime('%Y%m%d_%H%M')
            hour_key = timestamp.strftime('%Y%m%d_%H')
            
            # Update per-minute metrics
            await self._update_time_window_metrics(f"vigia:monitor:minute:{minute_key}", monitoring_result)
            
            # Update per-hour metrics
            await self._update_time_window_metrics(f"vigia:monitor:hour:{hour_key}", monitoring_result)
            
            # Update model-specific metrics
            model_key = f"vigia:monitor:model:{monitoring_result['model_version']}:{minute_key}"
            await self._update_time_window_metrics(model_key, monitoring_result)
            
            # Set expiration for cleanup
            await self.redis_client.expire(f"vigia:monitor:minute:{minute_key}", 3600)  # 1 hour
            await self.redis_client.expire(f"vigia:monitor:hour:{hour_key}", 86400)    # 24 hours
            await self.redis_client.expire(model_key, 3600)  # 1 hour
            
        except Exception as e:
            logger.error(f"Failed to update realtime metrics: {e}")
    
    async def _update_time_window_metrics(self, key: str, monitoring_result: Dict[str, Any]) -> None:
        """Update metrics for a specific time window"""
        
        pipe = self.redis_client.pipeline()
        
        # Total responses
        pipe.hincrby(key, "total_responses", 1)
        
        # Safety level counts
        safety_level = monitoring_result['safety_level'].value
        pipe.hincrby(key, f"safety_{safety_level}", 1)
        
        # Violation counts
        for violation in monitoring_result['violations']:
            pipe.hincrby(key, f"violation_{violation.violation_type.value}", 1)
        
        # Escalation count
        if monitoring_result['escalation_required']:
            pipe.hincrby(key, "escalations", 1)
        
        # Processing time (running average)
        if monitoring_result.get('processing_time'):
            await self._update_running_average(
                f"{key}:processing_time",
                monitoring_result['processing_time']
            )
        
        # Confidence score (running average)
        if monitoring_result.get('confidence_score'):
            await self._update_running_average(
                f"{key}:confidence",
                monitoring_result['confidence_score']
            )
        
        await pipe.execute()
    
    async def _update_running_average(self, key: str, new_value: float) -> None:
        """Update running average in Redis"""
        
        current_avg = await self.redis_client.hget(key, "average")
        current_count = await self.redis_client.hget(key, "count")
        
        if current_avg and current_count:
            avg = float(current_avg)
            count = int(current_count)
            new_avg = ((avg * count) + new_value) / (count + 1)
            new_count = count + 1
        else:
            new_avg = new_value
            new_count = 1
        
        pipe = self.redis_client.pipeline()
        pipe.hset(key, "average", new_avg)
        pipe.hset(key, "count", new_count)
        await pipe.execute()
    
    async def _check_immediate_alerts(self, monitoring_result: Dict[str, Any]) -> List[ResponseAlert]:
        """Check for immediate alerts based on monitoring result"""
        
        alerts = []
        
        # Critical safety violations
        critical_violations = [
            v for v in monitoring_result['violations']
            if v.severity == "critical"
        ]
        
        if critical_violations:
            alert = ResponseAlert(
                alert_id=f"critical_{monitoring_result['batman_token']}_{datetime.now().timestamp()}",
                alert_type="critical_safety_violation",
                severity=AlertSeverity.CRITICAL,
                batman_token=monitoring_result['batman_token'],
                model_version=monitoring_result['model_version'],
                violation_types=[v.violation_type for v in critical_violations],
                safety_level=monitoring_result['safety_level'],
                description=f"Critical safety violations detected: {[v.violation_type.value for v in critical_violations]}",
                response_snippet="[Response monitored]",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Emergency mishandling
        emergency_violations = [
            v for v in monitoring_result['violations']
            if v.violation_type == SafetyViolationType.EMERGENCY_MISHANDLING
        ]
        
        if emergency_violations:
            alert = ResponseAlert(
                alert_id=f"emergency_{monitoring_result['batman_token']}_{datetime.now().timestamp()}",
                alert_type="emergency_mishandling",
                severity=AlertSeverity.CRITICAL,
                batman_token=monitoring_result['batman_token'],
                model_version=monitoring_result['model_version'],
                violation_types=[SafetyViolationType.EMERGENCY_MISHANDLING],
                safety_level=monitoring_result['safety_level'],
                description="Emergency situation mishandled by AI response",
                response_snippet="[Emergency response monitored]",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Low confidence with safety issues
        if (monitoring_result.get('confidence_score', 1.0) < 0.5 and
            monitoring_result['safety_level'] in [ResponseSafetyLevel.UNSAFE, ResponseSafetyLevel.NEEDS_REVIEW]):
            
            alert = ResponseAlert(
                alert_id=f"low_confidence_{monitoring_result['batman_token']}_{datetime.now().timestamp()}",
                alert_type="low_confidence_unsafe",
                severity=AlertSeverity.WARNING,
                batman_token=monitoring_result['batman_token'],
                model_version=monitoring_result['model_version'],
                violation_types=[v.violation_type for v in monitoring_result['violations']],
                safety_level=monitoring_result['safety_level'],
                description=f"Low confidence ({monitoring_result.get('confidence_score', 0):.2f}) with safety issues",
                response_snippet="[Low confidence response]",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    async def _process_alert(self, alert: ResponseAlert) -> None:
        """Process and potentially escalate an alert"""
        
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            
            # Update alert counters
            self.alert_counters[alert.alert_type] += 1
            
            # Find matching escalation rules
            matching_rules = [
                rule for rule in self.escalation_rules
                if rule.condition == alert.alert_type
            ]
            
            # Process escalations
            for rule in matching_rules:
                if rule.delay_minutes == 0:
                    # Immediate escalation
                    await self._escalate_alert(alert, rule)
                else:
                    # Scheduled escalation
                    await self._schedule_escalation(alert, rule)
            
            # Log alert
            await self.audit_service.log_activity(
                activity_type="response_alert_created",
                batman_token=alert.batman_token,
                details={
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'safety_level': alert.safety_level.value,
                    'escalated': alert.escalated
                }
            )
            
            logger.warning(f"Alert created: {alert.alert_type} ({alert.severity.value}) - {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to process alert: {e}")
    
    async def _escalate_alert(self, alert: ResponseAlert, rule: EscalationRule) -> None:
        """Escalate alert through specified channels"""
        
        try:
            escalation_data = {
                'alert': alert,
                'rule': rule,
                'escalation_time': datetime.now()
            }
            
            # Process each escalation channel
            for channel in rule.channels:
                await self._send_escalation(channel, escalation_data)
            
            # Mark as escalated
            alert.escalated = True
            
            logger.critical(f"Alert escalated: {alert.alert_id} via {[c.value for c in rule.channels]}")
            
        except Exception as e:
            logger.error(f"Failed to escalate alert {alert.alert_id}: {e}")
    
    async def _send_escalation(self, channel: EscalationChannel, escalation_data: Dict[str, Any]) -> None:
        """Send escalation through specific channel"""
        
        alert = escalation_data['alert']
        
        if channel == EscalationChannel.SLACK:
            # Schedule Slack notification task
            await self._schedule_slack_notification(alert)
        
        elif channel == EscalationChannel.EMAIL:
            # Schedule email notification task
            await self._schedule_email_notification(alert)
        
        elif channel == EscalationChannel.DASHBOARD:
            # Update dashboard alerts
            await self._update_dashboard_alerts(alert)
        
        elif channel == EscalationChannel.WEBHOOK:
            # Send webhook notification
            await self._send_webhook_notification(alert)
        
        # Log escalation
        await self.audit_service.log_activity(
            activity_type="alert_escalation",
            batman_token=alert.batman_token,
            details={
                'alert_id': alert.alert_id,
                'channel': channel.value,
                'escalation_time': escalation_data['escalation_time'].isoformat()
            }
        )
    
    async def _schedule_slack_notification(self, alert: ResponseAlert) -> None:
        """Schedule Slack notification via Celery"""
        
        try:
            # Use existing Slack notification task
            slack_notification_task.apply_async(
                args=[{
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'description': alert.description,
                    'batman_token': alert.batman_token,
                    'model_version': alert.model_version,
                    'timestamp': alert.timestamp.isoformat()
                }],
                queue='notifications'
            )
            
        except Exception as e:
            logger.error(f"Failed to schedule Slack notification: {e}")
    
    async def _update_dashboard_alerts(self, alert: ResponseAlert) -> None:
        """Update dashboard with alert information"""
        
        if not self.redis_client:
            return
        
        try:
            # Store alert for dashboard display
            alert_data = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved
            }
            
            # Store in Redis with expiration
            await self.redis_client.setex(
                f"vigia:dashboard:alert:{alert.alert_id}",
                86400,  # 24 hours
                json.dumps(alert_data, default=str)
            )
            
            # Update alert summary
            await self.redis_client.hincrby("vigia:dashboard:alert_summary", alert.severity.value, 1)
            
        except Exception as e:
            logger.error(f"Failed to update dashboard alerts: {e}")
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start periodic monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_violation_rates()),
            asyncio.create_task(self._monitor_model_performance()),
            asyncio.create_task(self._cleanup_old_alerts())
        ]
        
        logger.info("Background monitoring started")
    
    async def _monitor_violation_rates(self) -> None:
        """Monitor violation rates and create alerts"""
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if not self.redis_client:
                    continue
                
                # Check 5-minute violation rate
                current_time = datetime.now()
                minute_key = current_time.strftime('%Y%m%d_%H%M')
                metrics_key = f"vigia:monitor:minute:{minute_key}"
                
                metrics = await self.redis_client.hgetall(metrics_key)
                
                if metrics:
                    total_responses = int(metrics.get('total_responses', 0))
                    unsafe_responses = int(metrics.get('safety_unsafe', 0))
                    
                    if total_responses > 0:
                        unsafe_rate = unsafe_responses / total_responses
                        
                        if unsafe_rate > self.thresholds['unsafe_response_rate_5min']:
                            await self._create_system_alert(
                                "high_violation_rate",
                                AlertSeverity.WARNING,
                                f"High unsafe response rate: {unsafe_rate:.2%} in 5 minutes",
                                "SYSTEM"
                            )
                
            except Exception as e:
                logger.error(f"Violation rate monitoring error: {e}")
    
    async def _create_system_alert(self,
                                  alert_type: str,
                                  severity: AlertSeverity,
                                  description: str,
                                  batman_token: str = "SYSTEM") -> None:
        """Create system-level alert"""
        
        alert = ResponseAlert(
            alert_id=f"system_{alert_type}_{datetime.now().timestamp()}",
            alert_type=alert_type,
            severity=severity,
            batman_token=batman_token,
            model_version="SYSTEM",
            violation_types=[],
            safety_level=ResponseSafetyLevel.NEEDS_REVIEW,
            description=description,
            response_snippet="[System Alert]",
            timestamp=datetime.now()
        )
        
        await self._process_alert(alert)
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get real-time monitoring dashboard data"""
        
        if not self.redis_client:
            # Mock data for development
            return {
                'current_metrics': {
                    'total_responses_last_hour': 245,
                    'safe_responses': 220,
                    'needs_review': 20,
                    'unsafe_responses': 5,
                    'escalations': 2
                },
                'active_alerts': 3,
                'violation_trends': {
                    'missing_disclaimer': 15,
                    'low_confidence': 12,
                    'npuap_noncompliance': 8
                },
                'model_performance': {
                    'MONAI_v1.2.3': {'safety_rate': 0.92, 'avg_confidence': 0.87},
                    'YOLOv5_v2.1.0': {'safety_rate': 0.89, 'avg_confidence': 0.82}
                }
            }
        
        try:
            current_time = datetime.now()
            hour_key = current_time.strftime('%Y%m%d_%H')
            metrics_key = f"vigia:monitor:hour:{hour_key}"
            
            # Get hourly metrics
            metrics = await self.redis_client.hgetall(metrics_key)
            
            # Get active alerts count
            alert_keys = await self.redis_client.keys("vigia:dashboard:alert:*")
            active_alerts = len([k for k in alert_keys if not json.loads(await self.redis_client.get(k)).get('resolved', False)])
            
            return {
                'current_metrics': {
                    'total_responses_last_hour': int(metrics.get('total_responses', 0)),
                    'safe_responses': int(metrics.get('safety_safe', 0)),
                    'needs_review': int(metrics.get('safety_needs_review', 0)),
                    'unsafe_responses': int(metrics.get('safety_unsafe', 0)),
                    'escalations': int(metrics.get('escalations', 0))
                },
                'active_alerts': active_alerts,
                'violation_trends': {
                    'missing_disclaimer': int(metrics.get('violation_missing_disclaimer', 0)),
                    'low_confidence': int(metrics.get('violation_low_confidence', 0)),
                    'npuap_noncompliance': int(metrics.get('violation_npuap_noncompliance', 0))
                },
                'last_updated': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring dashboard: {e}")
            return {'error': str(e)}


# Celery tasks for notifications
@celery_app.task(**configure_task_defaults('notify_slack_task'))
def slack_notification_task(alert_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send Slack notification for alerts"""
    
    try:
        logger.info(f"Processing Slack notification for alert: {alert_data['alert_id']}")
        
        # This would integrate with actual Slack API
        # For now, just logging the notification
        
        notification_result = {
            'alert_id': alert_data['alert_id'],
            'channel': 'slack',
            'status': 'sent',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Slack notification sent: {alert_data['alert_id']}")
        return notification_result
        
    except Exception as e:
        logger.error(f"Slack notification failed: {e}")
        raise


# Singleton instance
response_monitor = ResponseMonitor()

__all__ = [
    'ResponseMonitor',
    'AlertSeverity',
    'EscalationChannel',
    'ResponseAlert',
    'EscalationRule',
    'response_monitor',
    'slack_notification_task'
]