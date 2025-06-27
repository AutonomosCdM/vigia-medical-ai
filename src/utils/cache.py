"""
Cache Module

Provides secure caching functionality for tokens and medical data.
Implements encryption-at-rest for sensitive cached information.
"""

import json
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from src.security.encryption import EncryptionManager

logger = logging.getLogger(__name__)


class TokenCache:
    """
    Secure token cache with encryption at rest.
    
    Provides encrypted storage for authentication tokens,
    patient aliases, and session data.
    """
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        """Initialize token cache with encryption"""
        self.encryption_manager = encryption_manager or EncryptionManager()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._expiration: Dict[str, datetime] = {}
        logger.info("Token cache initialized with encryption")
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Set encrypted value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
        """
        try:
            # Encrypt data before storage
            encrypted_data = self._encrypt_before_storage(value)
            
            # Store encrypted data
            self._cache[key] = {
                'data': encrypted_data,
                'encrypted': True,
                'stored_at': datetime.utcnow().isoformat()
            }
            
            # Set expiration if specified
            if ttl_seconds:
                self._expiration[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            logger.debug(f"Token cached with encryption: {key}")
            
        except Exception as e:
            logger.error(f"Failed to cache token {key}: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get and decrypt value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Decrypted value or None if not found/expired
        """
        try:
            # Check if expired
            if self._is_expired(key):
                self.delete(key)
                return None
            
            # Get cached data
            cached_item = self._cache.get(key)
            if not cached_item:
                return None
            
            # Decrypt if encrypted
            if cached_item.get('encrypted', False):
                encrypted_data = cached_item['data']
                return self._decrypt_after_retrieval(encrypted_data)
            else:
                return cached_item['data']
                
        except Exception as e:
            logger.error(f"Failed to retrieve cached token {key}: {e}")
            return None
    
    def delete(self, key: str):
        """Delete cached item"""
        self._cache.pop(key, None)
        self._expiration.pop(key, None)
        logger.debug(f"Token removed from cache: {key}")
    
    def clear(self):
        """Clear all cached items"""
        self._cache.clear()
        self._expiration.clear()
        logger.info("Token cache cleared")
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        if self._is_expired(key):
            self.delete(key)
            return False
        return key in self._cache
    
    def keys(self) -> list:
        """Get all valid cache keys"""
        # Clean expired keys first
        self._cleanup_expired()
        return list(self._cache.keys())
    
    def size(self) -> int:
        """Get cache size"""
        self._cleanup_expired()
        return len(self._cache)
    
    def _encrypt_before_storage(self, data: Any) -> bytes:
        """
        Encrypt data before storing in cache
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data bytes
        """
        try:
            # Convert to JSON string if not already string
            if isinstance(data, (dict, list)):
                json_data = json.dumps(data, sort_keys=True)
            elif isinstance(data, str):
                json_data = data
            else:
                json_data = str(data)
            
            # Encrypt the data
            encrypted = self.encryption_manager.encrypt(json_data)
            return encrypted.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Cache encryption failed: {e}")
            raise
    
    def _decrypt_after_retrieval(self, encrypted_data: bytes) -> Any:
        """
        Decrypt data after retrieval from cache
        
        Args:
            encrypted_data: Encrypted data bytes
            
        Returns:
            Decrypted data
        """
        try:
            # Decrypt the data
            encrypted_str = encrypted_data.decode('utf-8')
            decrypted = self.encryption_manager.decrypt(encrypted_str)
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(decrypted)
            except json.JSONDecodeError:
                return decrypted
                
        except Exception as e:
            logger.error(f"Cache decryption failed: {e}")
            raise
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache key is expired"""
        expiration = self._expiration.get(key)
        if expiration and datetime.utcnow() > expiration:
            return True
        return False
    
    def _cleanup_expired(self):
        """Remove expired items from cache"""
        expired_keys = []
        for key in self._cache:
            if self._is_expired(key):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._cleanup_expired()
        
        total_items = len(self._cache)
        encrypted_items = sum(1 for item in self._cache.values() if item.get('encrypted', False))
        
        return {
            'total_items': total_items,
            'encrypted_items': encrypted_items,
            'expiring_items': len(self._expiration),
            'encryption_ratio': round(encrypted_items / total_items * 100, 2) if total_items > 0 else 0
        }


class MedicalDataCache(TokenCache):
    """
    Specialized cache for medical data with enhanced security.
    
    Extends TokenCache with medical-specific caching patterns
    and additional security measures.
    """
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        """Initialize medical data cache"""
        super().__init__(encryption_manager)
        self._access_log: Dict[str, list] = {}
        logger.info("Medical data cache initialized")
    
    def cache_patient_data(self, patient_id: str, data: Dict[str, Any], ttl_hours: int = 24):
        """
        Cache patient data with medical-grade security
        
        Args:
            patient_id: Patient identifier
            data: Patient data
            ttl_hours: Cache TTL in hours
        """
        cache_key = f"patient:{patient_id}"
        ttl_seconds = ttl_hours * 3600
        
        # Log access for audit
        self._log_access(cache_key, 'CACHE_WRITE')
        
        # Cache with encryption
        self.set(cache_key, data, ttl_seconds)
        logger.info(f"Patient data cached for {patient_id} (TTL: {ttl_hours}h)")
    
    def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached patient data
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Patient data or None
        """
        cache_key = f"patient:{patient_id}"
        
        # Log access for audit
        self._log_access(cache_key, 'CACHE_READ')
        
        data = self.get(cache_key)
        if data:
            logger.debug(f"Patient data retrieved from cache: {patient_id}")
        
        return data
    
    def cache_medical_result(self, result_id: str, result: Dict[str, Any], ttl_hours: int = 48):
        """
        Cache medical analysis result
        
        Args:
            result_id: Result identifier
            result: Medical result data
            ttl_hours: Cache TTL in hours
        """
        cache_key = f"result:{result_id}"
        ttl_seconds = ttl_hours * 3600
        
        self._log_access(cache_key, 'CACHE_WRITE')
        self.set(cache_key, result, ttl_seconds)
        logger.info(f"Medical result cached: {result_id}")
    
    def get_medical_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached medical result
        
        Args:
            result_id: Result identifier
            
        Returns:
            Medical result or None
        """
        cache_key = f"result:{result_id}"
        
        self._log_access(cache_key, 'CACHE_READ')
        return self.get(cache_key)
    
    def _log_access(self, key: str, operation: str):
        """Log cache access for audit trail"""
        if key not in self._access_log:
            self._access_log[key] = []
        
        self._access_log[key].append({
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat(),
            'user': 'system'  # Could be enhanced with actual user context
        })
        
        # Keep only recent access logs (last 100 per key)
        if len(self._access_log[key]) > 100:
            self._access_log[key] = self._access_log[key][-100:]
    
    def get_access_log(self, key: Optional[str] = None) -> Dict[str, list]:
        """
        Get cache access log for audit
        
        Args:
            key: Specific key to get log for, or None for all
            
        Returns:
            Access log entries
        """
        if key:
            return {key: self._access_log.get(key, [])}
        return self._access_log.copy()
    
    def clear_patient_cache(self, patient_id: str):
        """Clear all cached data for specific patient"""
        keys_to_remove = [key for key in self.keys() if key.startswith(f"patient:{patient_id}")]
        
        for key in keys_to_remove:
            self.delete(key)
            self._log_access(key, 'CACHE_DELETE')
        
        logger.info(f"Cleared cache for patient {patient_id} ({len(keys_to_remove)} items)")