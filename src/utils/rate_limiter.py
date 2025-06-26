"""
Rate limiting utilities for Vigia services.
Implements rate limiting using Redis as backend storage.
"""

import os
import time
import logging
from typing import Optional, Tuple
from functools import wraps
from flask import request, jsonify, g
from fastapi import HTTPException, Request
import redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter for API endpoints."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize rate limiter.
        
        Args:
            redis_url: Redis connection URL. Defaults to REDIS_URL env var.
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.enabled = os.getenv('RATE_LIMIT_ENABLED', 'false').lower() == 'true'
        self.default_limit = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
        
        if self.enabled:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Rate limiter connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for rate limiting: {e}")
                self.enabled = False
        else:
            logger.info("Rate limiting is disabled")
    
    def is_allowed(self, key: str, limit: Optional[int] = None, window: int = 60) -> Tuple[bool, dict]:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            key: Unique identifier for the rate limit (e.g., IP address, user ID)
            limit: Maximum number of requests per window. Defaults to configured limit.
            window: Time window in seconds. Defaults to 60 seconds.
            
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        if not self.enabled:
            return True, {"rate_limit_enabled": False}
        
        limit = limit or self.default_limit
        current_time = int(time.time())
        window_start = current_time - (current_time % window)
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            redis_key = f"rate_limit:{key}:{window_start}"
            
            # Get current count
            pipe.get(redis_key)
            # Increment counter
            pipe.incr(redis_key)
            # Set expiration
            pipe.expire(redis_key, window * 2)  # Keep for 2 windows for safety
            
            results = pipe.execute()
            current_count = int(results[1])  # Count after increment
            
            is_allowed = current_count <= limit
            
            info = {
                "rate_limit_enabled": True,
                "limit": limit,
                "window": window,
                "current_count": current_count,
                "remaining": max(0, limit - current_count),
                "reset_time": window_start + window,
                "allowed": is_allowed
            }
            
            if not is_allowed:
                logger.warning(f"Rate limit exceeded for key {key}: {current_count}/{limit}")
            
            return is_allowed, info
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Fail open - allow request if Redis is down
            return True, {"rate_limit_enabled": False, "error": str(e)}
    
    def get_client_key(self, request_obj) -> str:
        """
        Extract client identifier from request.
        
        Args:
            request_obj: Flask request or FastAPI Request object
            
        Returns:
            String identifier for the client
        """
        # Try to get real IP from headers (for reverse proxy setups)
        real_ip = None
        
        if hasattr(request_obj, 'headers'):  # FastAPI Request
            real_ip = (
                request_obj.headers.get('X-Forwarded-For', '').split(',')[0].strip() or
                request_obj.headers.get('X-Real-IP') or
                request_obj.client.host if request_obj.client else None
            )
        elif hasattr(request_obj, 'environ'):  # Flask request
            real_ip = (
                request_obj.environ.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip() or
                request_obj.environ.get('HTTP_X_REAL_IP') or
                request_obj.remote_addr
            )
        
        return real_ip or 'unknown'


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit_flask(limit: Optional[int] = None, window: int = 60, key_func: Optional[callable] = None):
    """
    Flask decorator for rate limiting.
    
    Args:
        limit: Maximum requests per window
        window: Time window in seconds
        key_func: Function to generate rate limit key. Defaults to client IP.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not rate_limiter.enabled:
                return f(*args, **kwargs)
            
            # Generate rate limit key
            if key_func:
                key = key_func()
            else:
                key = rate_limiter.get_client_key(request)
            
            # Check rate limit
            is_allowed, info = rate_limiter.is_allowed(key, limit, window)
            
            # Add rate limit headers to response
            g.rate_limit_info = info
            
            if not is_allowed:
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {info['limit']} per {info['window']} seconds",
                    "retry_after": info['reset_time'] - int(time.time())
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(info['reset_time'] - int(time.time()))
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset_time'])
                return response
            
            # Call original function
            result = f(*args, **kwargs)
            
            # Add rate limit headers to successful responses
            if hasattr(result, 'headers') and hasattr(g, 'rate_limit_info'):
                info = g.rate_limit_info
                result.headers['X-RateLimit-Limit'] = str(info['limit'])
                result.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                result.headers['X-RateLimit-Reset'] = str(info['reset_time'])
            
            return result
        return decorated_function
    return decorator


def rate_limit_fastapi(limit: Optional[int] = None, window: int = 60, key_func: Optional[callable] = None):
    """
    FastAPI dependency for rate limiting.
    
    Args:
        limit: Maximum requests per window
        window: Time window in seconds
        key_func: Function to generate rate limit key. Defaults to client IP.
    """
    async def dependency(request: Request):
        if not rate_limiter.enabled:
            return True
        
        # Generate rate limit key
        if key_func:
            key = key_func(request)
        else:
            key = rate_limiter.get_client_key(request)
        
        # Check rate limit
        is_allowed, info = rate_limiter.is_allowed(key, limit, window)
        
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {info['limit']} per {info['window']} seconds",
                    "retry_after": info['reset_time'] - int(time.time()),
                    "rate_limit": info
                },
                headers={
                    "Retry-After": str(info['reset_time'] - int(time.time())),
                    "X-RateLimit-Limit": str(info['limit']),
                    "X-RateLimit-Remaining": str(info['remaining']),
                    "X-RateLimit-Reset": str(info['reset_time'])
                }
            )
        
        # Store rate limit info for response headers
        request.state.rate_limit_info = info
        return True
    
    return dependency


def add_rate_limit_headers(response, request):
    """
    Add rate limit headers to FastAPI response.
    
    Args:
        response: FastAPI response object
        request: FastAPI request object
    """
    if hasattr(request.state, 'rate_limit_info'):
        info = request.state.rate_limit_info
        response.headers['X-RateLimit-Limit'] = str(info['limit'])
        response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
        response.headers['X-RateLimit-Reset'] = str(info['reset_time'])
    return response