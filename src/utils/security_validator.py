"""
Security validation and sanitization utilities for Vigia
Provides input validation, sanitization, and security checks
"""

import re
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import numpy as np

# Try to import magic, fallback if not available
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    magic = None

logger = logging.getLogger(__name__)

# Security constants
MAX_IMAGE_SIZE_MB = 50
MAX_TEXT_LENGTH = 10000
MAX_FILENAME_LENGTH = 255
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/png', 
    'image/bmp',
    'image/tiff'
}

# Dangerous patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE)\b)",
    r"(--|#|\/\*|\*\/)",
    r"(\bOR\b\s*\d+\s*=\s*\d+)",
    r"('\s*OR\s*')",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e",
    r"%252e%252e",
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"onerror\s*=",
    r"onload\s*=",
    r"onclick\s*=",
]


class SecurityValidator:
    """Comprehensive security validation for all inputs"""
    
    def __init__(self):
        """Initialize security validator"""
        if HAS_MAGIC:
            self.mime = magic.Magic(mime=True)
        else:
            self.mime = None
            logger.warning("python-magic not available, MIME type validation disabled")
    
    def validate_image(self, 
                      file_path: str, 
                      max_size_mb: int = MAX_IMAGE_SIZE_MB) -> Tuple[bool, Optional[str]]:
        """
        Validate image file for security risks
        
        Args:
            file_path: Path to image file
            max_size_mb: Maximum allowed file size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check file exists
            if not path.exists():
                return False, "File does not exist"
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
            
            # Check extension
            if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                return False, f"Invalid extension: {path.suffix}"
            
            # Check MIME type (if magic is available)
            if self.mime:
                mime_type = self.mime.from_file(str(path))
                if mime_type not in ALLOWED_MIME_TYPES:
                    return False, f"Invalid MIME type: {mime_type}"
            else:
                logger.debug("Skipping MIME type validation (python-magic not available)")
            
            # Try to open with PIL to verify it's a valid image
            try:
                with Image.open(path) as img:
                    # Check image dimensions
                    width, height = img.size
                    if width > 10000 or height > 10000:
                        return False, f"Image dimensions too large: {width}x{height}"
                    
                    # Check for potential image bombs
                    pixels = width * height
                    if pixels > 100_000_000:  # 100 megapixels
                        return False, "Image has too many pixels (potential decompression bomb)"
                    
            except Exception as e:
                return False, f"Invalid image file: {str(e)}"
            
            # Check filename for path traversal
            if self._contains_path_traversal(str(path)):
                return False, "Filename contains path traversal attempt"
            
            logger.info(f"Image validation passed: {path.name}")
            return True, None
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def sanitize_text(self, 
                     text: str, 
                     max_length: int = MAX_TEXT_LENGTH,
                     allow_unicode: bool = True) -> str:
        """
        Sanitize text input by removing dangerous patterns
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            allow_unicode: Whether to allow unicode characters
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters (except newline and tab)
        text = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Encode/decode to handle unicode properly
        if not allow_unicode:
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove SQL injection attempts
        for pattern in SQL_INJECTION_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove XSS attempts
        for pattern in XSS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Escape HTML entities
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and other attacks
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed"
        
        # Get base name only (remove any path)
        filename = os.path.basename(filename)
        
        # Remove null bytes
        filename = filename.replace('\x00', '')
        
        # Replace dangerous characters
        filename = re.sub(r'[^\w\s.-]', '_', filename)
        
        # Remove multiple dots (prevent extension spoofing)
        filename = re.sub(r'\.{2,}', '.', filename)
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(filename) > MAX_FILENAME_LENGTH:
            max_name_length = MAX_FILENAME_LENGTH - len(ext)
            name = name[:max_name_length]
            filename = name + ext
        
        # Ensure has extension
        if not ext and '.' not in filename:
            filename += '.unknown'
        
        return filename
    
    def validate_patient_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate patient code format and safety
        
        Args:
            code: Patient code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Patient code is required"
        
        # Check format (XX-YYYY-NNN)
        pattern = r'^[A-Z]{2}-\d{4}-\d{3}$'
        if not re.match(pattern, code):
            return False, "Invalid format. Expected: XX-YYYY-NNN"
        
        # Additional safety checks
        if self._contains_sql_injection(code):
            return False, "Invalid characters detected"
        
        return True, None
    
    def validate_webhook_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook URL for safety
        
        Args:
            url: Webhook URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url:
            return False, "URL is required"
        
        # Basic URL pattern
        url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^#\s]*)?$'
        if not re.match(url_pattern, url):
            return False, "Invalid URL format"
        
        # Check for localhost/internal IPs (SSRF prevention)
        dangerous_hosts = [
            'localhost', '127.0.0.1', '0.0.0.0',
            '169.254.169.254',  # AWS metadata
            '::1', '[::1]'
        ]
        
        for host in dangerous_hosts:
            if host in url.lower():
                return False, "Internal URLs not allowed"
        
        # Check for private IP ranges
        private_ip_patterns = [
            r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            r'172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}',
            r'192\.168\.\d{1,3}\.\d{1,3}'
        ]
        
        for pattern in private_ip_patterns:
            if re.search(pattern, url):
                return False, "Private IP addresses not allowed"
        
        return True, None
    
    def _contains_sql_injection(self, text: str) -> bool:
        """Check if text contains SQL injection patterns"""
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _contains_path_traversal(self, text: str) -> bool:
        """Check if text contains path traversal patterns"""
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """
        Hash sensitive data for logging
        
        Args:
            data: Sensitive data to hash
            salt: Optional salt for hashing
            
        Returns:
            Hashed string
        """
        if salt:
            data = f"{salt}{data}"
        
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def mask_sensitive_data(self, data: str, visible_chars: int = 4) -> str:
        """
        Mask sensitive data for logging
        
        Args:
            data: Sensitive data to mask
            visible_chars: Number of characters to leave visible
            
        Returns:
            Masked string
        """
        if len(data) <= visible_chars * 2:
            return "*" * len(data)
        
        return f"{data[:visible_chars]}{'*' * (len(data) - visible_chars * 2)}{data[-visible_chars:]}"


# Global validator instance
security_validator = SecurityValidator()


def validate_and_sanitize_image(file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate and sanitize image file
    
    Returns:
        Tuple of (is_valid, error_message, sanitized_filename)
    """
    validator = security_validator
    
    # Validate image
    is_valid, error = validator.validate_image(file_path)
    if not is_valid:
        return False, error, None
    
    # Sanitize filename
    filename = os.path.basename(file_path)
    sanitized_filename = validator.sanitize_filename(filename)
    
    return True, None, sanitized_filename


def sanitize_user_input(text: str, input_type: str = "general") -> str:
    """
    Sanitize user input based on type
    
    Args:
        text: Input text
        input_type: Type of input (general, patient_code, filename)
        
    Returns:
        Sanitized text
    """
    validator = security_validator
    
    if input_type == "patient_code":
        # For patient codes, validate format
        is_valid, _ = validator.validate_patient_code(text)
        if is_valid:
            return text
        else:
            return ""
    
    elif input_type == "filename":
        return validator.sanitize_filename(text)
    
    else:
        return validator.sanitize_text(text)