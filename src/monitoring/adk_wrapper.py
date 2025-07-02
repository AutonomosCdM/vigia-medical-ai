"""
VIGIA Medical AI - ADK (Agent Development Kit) Wrapper
=====================================================

Production ADK wrapper for agent monitoring with medical telemetry integration.
Provides decorator-based monitoring for medical AI agents.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from .medical_telemetry import MedicalTelemetry
from ..utils.secure_logger import SecureLogger

logger = SecureLogger(__name__)

# Global telemetry instance for ADK
_global_telemetry: Optional[MedicalTelemetry] = None

def initialize_adk_telemetry(app_id: str = "vigia-adk", environment: str = "production") -> MedicalTelemetry:
    """Initialize global ADK telemetry"""
    global _global_telemetry
    _global_telemetry = MedicalTelemetry(
        app_id=app_id,
        environment=environment,
        enable_phi_protection=True
    )
    logger.info(f"ADK telemetry initialized: {app_id} ({environment})")
    return _global_telemetry

def get_adk_telemetry() -> Optional[MedicalTelemetry]:
    """Get global ADK telemetry instance"""
    return _global_telemetry

def adk_agent_wrapper(agent_type: str = None, 
                     track_performance: bool = True,
                     record_events: bool = True):
    """
    Production ADK agent wrapper with medical telemetry
    
    Args:
        agent_type: Type of medical agent (e.g., "image_analysis", "risk_assessment")
        track_performance: Whether to track performance metrics
        record_events: Whether to record agent events
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get telemetry instance
            telemetry = get_adk_telemetry()
            if not telemetry:
                logger.warning("ADK telemetry not initialized, running without monitoring")
                return await func(*args, **kwargs)
            
            # Extract Batman token from arguments
            batman_token = _extract_batman_token(args, kwargs)
            if not batman_token:
                logger.warning("No Batman token found in agent call")
                return await func(*args, **kwargs)
            
            # Performance tracking
            start_time = time.time()
            function_name = func.__name__
            agent_name = agent_type or function_name
            
            # Start session if needed
            session_id = f"adk_{agent_name}_{batman_token}_{int(start_time)}"
            
            try:
                # Record agent start event
                if record_events and telemetry:
                    await telemetry.record_medical_event(
                        session_id=session_id,
                        event_type=f"agent_{function_name}_start",
                        event_data={
                            "agent_type": agent_name,
                            "function_name": function_name,
                            "batman_token": batman_token,
                            "start_time": start_time,
                            "args_count": len(args),
                            "kwargs_count": len(kwargs)
                        },
                        agent_action=True
                    )
                
                # Execute the wrapped function
                logger.debug(f"ADK executing: {agent_name}.{function_name}")
                result = await func(*args, **kwargs)
                
                # Calculate performance metrics
                end_time = time.time()
                duration = end_time - start_time
                
                # Record success event
                if record_events and telemetry:
                    await telemetry.record_medical_event(
                        session_id=session_id,
                        event_type=f"agent_{function_name}_success",
                        event_data={
                            "agent_type": agent_name,
                            "function_name": function_name,
                            "batman_token": batman_token,
                            "duration_seconds": duration,
                            "result_type": type(result).__name__ if result else "None",
                            "performance_good": duration < 5.0  # Medical response time threshold
                        },
                        agent_action=True
                    )
                
                # Log performance
                if track_performance:
                    if duration > 5.0:
                        logger.warning(f"ADK performance alert: {agent_name}.{function_name} took {duration:.2f}s")
                    else:
                        logger.debug(f"ADK performance: {agent_name}.{function_name} completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                # Calculate error timing
                end_time = time.time()
                duration = end_time - start_time
                
                # Record error event
                if record_events and telemetry:
                    await telemetry.record_medical_event(
                        session_id=session_id,
                        event_type=f"agent_{function_name}_error",
                        event_data={
                            "agent_type": agent_name,
                            "function_name": function_name,
                            "batman_token": batman_token,
                            "duration_seconds": duration,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:200]  # Truncate for safety
                        },
                        agent_action=True
                    )
                
                logger.error(f"ADK error in {agent_name}.{function_name}: {e}")
                raise  # Re-raise the original exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, run without telemetry (simplified)
            logger.debug(f"ADK sync wrapper: {func.__name__}")
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def _extract_batman_token(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract Batman token from function arguments"""
    
    # Check common argument names
    for key in ['batman_token', 'token_id', 'patient_token']:
        if key in kwargs:
            token = kwargs[key]
            if isinstance(token, str) and token.startswith('batman_'):
                return token
    
    # Check positional arguments for Batman tokens
    for arg in args:
        if isinstance(arg, str) and arg.startswith('batman_'):
            return arg
        elif isinstance(arg, dict):
            # Check if it's a context dict with Batman token
            for key in ['batman_token', 'token_id', 'patient_token']:
                if key in arg and isinstance(arg[key], str) and arg[key].startswith('batman_'):
                    return arg[key]
    
    return None

def adk_medical_agent(agent_type: str):
    """Specialized wrapper for medical agents with standard configuration"""
    return adk_agent_wrapper(
        agent_type=agent_type,
        track_performance=True,
        record_events=True
    )

def adk_performance_critical(agent_type: str, max_duration: float = 3.0):
    """Wrapper for performance-critical medical agents"""
    def decorator(func: Callable) -> Callable:
        base_wrapper = adk_agent_wrapper(
            agent_type=agent_type,
            track_performance=True,
            record_events=True
        )
        
        @wraps(func)
        async def critical_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await base_wrapper(func)(*args, **kwargs)
            duration = time.time() - start_time
            
            if duration > max_duration:
                logger.critical(f"Performance violation: {agent_type}.{func.__name__} took {duration:.2f}s (max: {max_duration}s)")
                
                # Record critical performance event
                telemetry = get_adk_telemetry()
                if telemetry:
                    batman_token = _extract_batman_token(args, kwargs)
                    if batman_token:
                        await telemetry.record_medical_event(
                            session_id=f"critical_{agent_type}_{int(start_time)}",
                            event_type="performance_violation",
                            event_data={
                                "agent_type": agent_type,
                                "function_name": func.__name__,
                                "duration_seconds": duration,
                                "max_allowed": max_duration,
                                "severity": "critical"
                            }
                        )
            
            return result
        
        return critical_wrapper
    return decorator

def get_adk_statistics() -> Dict[str, Any]:
    """Get ADK monitoring statistics"""
    telemetry = get_adk_telemetry()
    if not telemetry:
        return {"status": "not_initialized"}
    
    return {
        "status": "active",
        "session_statistics": telemetry.get_session_statistics(),
        "monitoring_active": True,
        "telemetry_app_id": telemetry.app_id,
        "environment": telemetry.environment
    }

__all__ = [
    'adk_agent_wrapper', 
    'adk_medical_agent', 
    'adk_performance_critical',
    'initialize_adk_telemetry',
    'get_adk_telemetry',
    'get_adk_statistics'
]