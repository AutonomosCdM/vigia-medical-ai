"""
PHI Tokenization Client
Cliente para integrar el servicio de tokenización con el pipeline de Vigia

Este cliente:
1. Se conecta al PHI Tokenization Service
2. Maneja la conversión Bruce Wayne → Batman
3. Proporciona datos tokenizados al resto del sistema
4. Mantiene cache de tokens válidos
5. Maneja errores y reintentos
"""

import os
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager

from ..utils.secure_logger import SecureLogger

logger = SecureLogger("phi_tokenization_client")

# ===============================================
# 1. CONFIGURATION
# ===============================================

@dataclass
class TokenizationClientConfig:
    """Configuration for PHI Tokenization Client"""
    
    # Service endpoint
    tokenization_service_url: str = os.getenv(
        "PHI_TOKENIZATION_SERVICE_URL", 
        "http://localhost:8080"
    )
    
    # Authentication
    staff_id: str = os.getenv("HOSPITAL_STAFF_ID", "VIGIA_SYSTEM")
    authorization_level: str = os.getenv("HOSPITAL_AUTH_LEVEL", "administrator")
    
    # Cache settings
    token_cache_ttl_minutes: int = int(os.getenv("TOKEN_CACHE_TTL_MINUTES", "30"))
    max_cache_size: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    
    # Request settings
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay_seconds: float = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))
    
    # System identification
    requesting_system: str = "vigia_lpp_detection"
    
    def __post_init__(self):
        if not self.tokenization_service_url.startswith(('http://', 'https://')):
            self.tokenization_service_url = f"http://{self.tokenization_service_url}"

# ===============================================
# 2. DATA MODELS
# ===============================================

@dataclass
class TokenizationRequest:
    """Request for patient tokenization"""
    hospital_mrn: str
    request_purpose: str
    urgency_level: str = "routine"
    hipaa_authorization: bool = True
    consent_form_signed: bool = True

@dataclass  
class TokenizedPatient:
    """Tokenized patient data (NO PHI)"""
    token_id: str
    patient_alias: str
    age_range: str
    gender_category: str
    risk_factors: Dict[str, Any]
    medical_conditions: Dict[str, Any]
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "patient_alias": self.patient_alias,
            "age_range": self.age_range,
            "gender_category": self.gender_category,
            "risk_factors": self.risk_factors,
            "medical_conditions": self.medical_conditions,
            "expires_at": self.expires_at.isoformat()
        }

@dataclass
class CachedToken:
    """Cached token information"""
    tokenized_patient: TokenizedPatient
    cached_at: datetime
    access_count: int = 0
    
    def is_expired(self, ttl_minutes: int) -> bool:
        """Check if cache entry is expired"""
        cache_age = datetime.now(timezone.utc) - self.cached_at
        return cache_age > timedelta(minutes=ttl_minutes)

# ===============================================
# 3. TOKEN CACHE
# ===============================================

class TokenCache:
    """In-memory cache for tokenized patients"""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.cache: Dict[str, CachedToken] = {}
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[TokenizedPatient]:
        """Get tokenized patient from cache"""
        if key not in self.cache:
            return None
        
        cached_token = self.cache[key]
        
        # Check if expired
        if cached_token.is_expired(self.ttl_minutes):
            self.remove(key)
            return None
        
        # Update access tracking
        cached_token.access_count += 1
        self._update_access_order(key)
        
        logger.audit("token_cache_hit", {
            "cache_key": key[:8] + "...",
            "access_count": cached_token.access_count,
            "cache_age_minutes": (datetime.now(timezone.utc) - cached_token.cached_at).total_seconds() / 60
        })
        
        return cached_token.tokenized_patient
    
    def put(self, key: str, tokenized_patient: TokenizedPatient):
        """Store tokenized patient in cache"""
        # Remove oldest entries if cache is full
        while len(self.cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            self.cache.pop(oldest_key, None)
        
        cached_token = CachedToken(
            tokenized_patient=tokenized_patient,
            cached_at=datetime.now(timezone.utc)
        )
        
        self.cache[key] = cached_token
        self._update_access_order(key)
        
        logger.audit("token_cache_store", {
            "cache_key": key[:8] + "...",
            "patient_alias": tokenized_patient.patient_alias,
            "cache_size": len(self.cache)
        })
    
    def remove(self, key: str):
        """Remove entry from cache"""
        self.cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _update_access_order(self, key: str):
        """Update access order for LRU eviction"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear_expired(self):
        """Clear all expired entries"""
        expired_keys = [
            key for key, cached_token in self.cache.items()
            if cached_token.is_expired(self.ttl_minutes)
        ]
        
        for key in expired_keys:
            self.remove(key)
        
        if expired_keys:
            logger.audit("token_cache_cleanup", {
                "expired_count": len(expired_keys),
                "remaining_count": len(self.cache)
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl_minutes,
            "total_access_count": sum(ct.access_count for ct in self.cache.values())
        }

# ===============================================
# 4. PHI TOKENIZATION CLIENT
# ===============================================

class PHITokenizationClient:
    """Client for PHI Tokenization Service"""
    
    def __init__(self, config: Optional[TokenizationClientConfig] = None):
        self.config = config or TokenizationClientConfig()
        self.cache = TokenCache(
            max_size=self.config.max_cache_size,
            ttl_minutes=self.config.token_cache_ttl_minutes
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.auth_expires_at: Optional[datetime] = None
        
        logger.audit("phi_tokenization_client_initialized", {
            "service_url": self.config.tokenization_service_url,
            "requesting_system": self.config.requesting_system,
            "cache_ttl_minutes": self.config.token_cache_ttl_minutes
        })
    
    async def initialize(self):
        """Initialize HTTP session and authenticate"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds),
            headers={"Content-Type": "application/json"}
        )
        
        # Authenticate with tokenization service
        await self._authenticate()
        
        logger.audit("phi_tokenization_client_ready", {
            "authenticated": self.auth_token is not None,
            "service_url": self.config.tokenization_service_url
        })
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def tokenize_patient(self, hospital_mrn: str, request_purpose: str, urgency_level: str = "routine") -> TokenizedPatient:
        """
        Tokenize patient data (Bruce Wayne → Batman)
        
        Args:
            hospital_mrn: Hospital Medical Record Number (e.g., "MRN-2025-001-BW")
            request_purpose: Purpose of tokenization (e.g., "LPP detection and analysis")
            urgency_level: Urgency level ("routine", "urgent", "emergency")
            
        Returns:
            TokenizedPatient object with no PHI
        """
        # Check cache first
        cache_key = f"mrn:{hospital_mrn}"
        cached_patient = self.cache.get(cache_key)
        if cached_patient:
            return cached_patient
        
        # Ensure we're authenticated
        await self._ensure_authenticated()
        
        request_data = {
            "hospital_mrn": hospital_mrn,
            "requesting_system": self.config.requesting_system,
            "request_purpose": request_purpose,
            "requested_by": self.config.staff_id,
            "authorization_level": self.config.authorization_level,
            "hipaa_authorization": True,
            "consent_form_signed": True,
            "urgency_level": urgency_level
        }
        
        logger.audit("tokenization_request_started", {
            "hospital_mrn": hospital_mrn[:8] + "...",  # Partial MRN for audit
            "request_purpose": request_purpose,
            "urgency_level": urgency_level
        })
        
        # Make tokenization request with retries
        response_data = await self._make_request_with_retries(
            "POST", "/tokenize", request_data
        )
        
        if not response_data["success"]:
            raise Exception(f"Tokenization failed: {response_data.get('message', 'Unknown error')}")
        
        # Create TokenizedPatient object
        tokenized_patient = TokenizedPatient(
            token_id=response_data["token_id"],
            patient_alias=response_data["patient_alias"],
            age_range=response_data["tokenized_data"]["age_range"],
            gender_category=response_data["tokenized_data"]["gender_category"],
            risk_factors=response_data["tokenized_data"]["risk_factors"],
            medical_conditions=response_data["tokenized_data"]["medical_conditions"],
            expires_at=datetime.fromisoformat(response_data["expires_at"].replace('Z', '+00:00'))
        )
        
        # Cache the result
        self.cache.put(cache_key, tokenized_patient)
        
        logger.audit("tokenization_completed", {
            "hospital_mrn": hospital_mrn[:8] + "...",
            "token_id": tokenized_patient.token_id,
            "patient_alias": tokenized_patient.patient_alias,
            "expires_at": tokenized_patient.expires_at.isoformat()
        })
        
        return tokenized_patient
    
    async def validate_token(self, token_id: str) -> Dict[str, Any]:
        """Validate existing token"""
        await self._ensure_authenticated()
        
        request_data = {
            "token_id": token_id,
            "requesting_system": self.config.requesting_system
        }
        
        response_data = await self._make_request_with_retries(
            "POST", "/validate", request_data
        )
        
        logger.audit("token_validation", {
            "token_id": token_id,
            "valid": response_data["valid"],
            "status": response_data.get("status", "unknown")
        })
        
        return response_data
    
    async def create_token_async(self, hospital_mrn: str, patient_data: dict = None) -> str:
        """
        Compatibility wrapper for Hume AI integration.
        Creates Batman token from hospital MRN.
        """
        try:
            # Initialize session if not done
            if not self.session:
                await self.initialize()
            
            # Use tokenize_patient method
            tokenized_patient = await self.tokenize_patient(
                hospital_mrn=hospital_mrn,
                request_purpose="voice_analysis_hume_ai",
                urgency_level="routine"
            )
            
            return tokenized_patient.token_id
            
        except Exception as e:
            # If service fails, create mock token for development
            import hashlib
            import time
            timestamp = int(time.time())
            mock_token = f"BATMAN_MOCK_{hashlib.md5(f'{hospital_mrn}_{timestamp}'.encode()).hexdigest()[:12]}"
            logger.warning(f"Using mock token {mock_token} due to service error: {e}")
            return mock_token
    
    async def get_tokenized_patient_by_token(self, token_id: str) -> Optional[TokenizedPatient]:
        """Get tokenized patient data by token ID"""
        # Check if token is valid first
        validation_result = await self.validate_token(token_id)
        
        if not validation_result["valid"]:
            return None
        
        # Try to find in cache
        for cached_token in self.cache.cache.values():
            if cached_token.tokenized_patient.token_id == token_id:
                return cached_token.tokenized_patient
        
        # If not in cache, we'd need to reconstruct from processing database
        # For now, return basic info from validation
        return TokenizedPatient(
            token_id=validation_result["token_id"],
            patient_alias=validation_result["patient_alias"],
            age_range="unknown",
            gender_category="unknown",
            risk_factors={},
            medical_conditions={},
            expires_at=datetime.fromisoformat(validation_result["expires_at"].replace('Z', '+00:00'))
        )
    
    async def _authenticate(self):
        """Authenticate with tokenization service"""
        try:
            login_data = {
                "staff_id": self.config.staff_id,
                "authorization_level": self.config.authorization_level
            }
            
            async with self.session.post(
                f"{self.config.tokenization_service_url}/auth/login",
                data=login_data
            ) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    self.auth_token = auth_data["access_token"]
                    self.auth_expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
                    
                    # Update session headers
                    self.session.headers.update({
                        "Authorization": f"Bearer {self.auth_token}"
                    })
                    
                    logger.audit("authentication_successful", {
                        "staff_id": self.config.staff_id,
                        "authorization_level": self.config.authorization_level
                    })
                else:
                    raise Exception(f"Authentication failed: {response.status}")
                    
        except Exception as e:
            logger.error("authentication_failed", {"error": str(e)})
            raise
    
    async def _ensure_authenticated(self):
        """Ensure we have valid authentication"""
        if not self.auth_token or not self.auth_expires_at:
            await self._authenticate()
            return
        
        # Check if token is about to expire (refresh 10 minutes early)
        if datetime.now(timezone.utc) + timedelta(minutes=10) >= self.auth_expires_at:
            await self._authenticate()
    
    async def _make_request_with_retries(self, method: str, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                url = f"{self.config.tokenization_service_url}{endpoint}"
                
                async with self.session.request(method, url, json=data) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return response_data
                    else:
                        raise Exception(f"HTTP {response.status}: {response_data.get('detail', 'Unknown error')}")
                        
            except Exception as e:
                last_error = e
                
                if attempt < self.config.max_retries:
                    logger.warning("request_retry", {
                        "attempt": attempt + 1,
                        "max_retries": self.config.max_retries,
                        "error": str(e),
                        "endpoint": endpoint
                    })
                    
                    await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                else:
                    logger.error("request_failed_all_retries", {
                        "attempts": attempt + 1,
                        "endpoint": endpoint,
                        "final_error": str(e)
                    })
        
        raise last_error
    
    async def cleanup_cache(self):
        """Cleanup expired cache entries"""
        self.cache.clear_expired()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

# ===============================================
# 5. CONVENIENCE FUNCTIONS
# ===============================================

# Global client instance
_client_instance: Optional[PHITokenizationClient] = None

async def get_tokenization_client() -> PHITokenizationClient:
    """Get or create tokenization client instance"""
    global _client_instance
    
    if _client_instance is None:
        _client_instance = PHITokenizationClient()
        await _client_instance.initialize()
    
    return _client_instance

async def tokenize_patient_phi(hospital_mrn: str, request_purpose: str) -> TokenizedPatient:
    """
    Convenience function to tokenize patient PHI
    
    Example:
        # Convert Bruce Wayne to Batman
        tokenized = await tokenize_patient_phi("MRN-2025-001-BW", "LPP detection and analysis")
        print(f"Patient tokenized as: {tokenized.patient_alias}")  # "Batman"
    """
    client = await get_tokenization_client()
    return await client.tokenize_patient(hospital_mrn, request_purpose)

async def get_patient_by_token(token_id: str) -> Optional[TokenizedPatient]:
    """Get tokenized patient by token ID"""
    client = await get_tokenization_client()
    return await client.get_tokenized_patient_by_token(token_id)

async def validate_patient_token(token_id: str) -> bool:
    """Validate if token is still valid"""
    client = await get_tokenization_client()
    result = await client.validate_token(token_id)
    return result["valid"]

# ===============================================
# 6. CONTEXT MANAGER
# ===============================================

@asynccontextmanager
async def tokenization_client():
    """Context manager for tokenization client"""
    client = PHITokenizationClient()
    try:
        await client.initialize()
        yield client
    finally:
        await client.close()

# ===============================================
# 7. EXAMPLE USAGE
# ===============================================

async def example_bruce_wayne_tokenization():
    """Example of tokenizing Bruce Wayne to Batman"""
    
    async with tokenization_client() as client:
        # Tokenize Bruce Wayne
        tokenized_patient = await client.tokenize_patient(
            hospital_mrn="MRN-2025-001-BW",
            request_purpose="Pressure injury detection and analysis",
            urgency_level="urgent"
        )
        
        print(f"Patient tokenized successfully:")
        print(f"  Original: Bruce Wayne (MRN-2025-001-BW)")
        print(f"  Tokenized: {tokenized_patient.patient_alias}")
        print(f"  Token ID: {tokenized_patient.token_id}")
        print(f"  Age Range: {tokenized_patient.age_range}")
        print(f"  Gender: {tokenized_patient.gender_category}")
        print(f"  Expires: {tokenized_patient.expires_at}")
        
        # Validate token
        is_valid = await client.validate_token(tokenized_patient.token_id)
        print(f"  Token Valid: {is_valid['valid']}")
        
        return tokenized_patient

if __name__ == "__main__":
    # Test the client
    asyncio.run(example_bruce_wayne_tokenization())