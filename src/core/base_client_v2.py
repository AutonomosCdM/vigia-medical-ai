"""
Enhanced base client using centralized configuration
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from config.settings import settings


class BaseClientV2(ABC):
    """
    Enhanced base client that uses centralized settings.
    All clients should inherit from this class.
    """
    
    def __init__(self, service_name: str, required_fields: List[str], 
                 optional_fields: Optional[List[str]] = None):
        """
        Initialize the base client.
        
        Args:
            service_name: Name of the service (for logging)
            required_fields: List of required setting field names
            optional_fields: List of optional setting field names
        """
        self.service_name = service_name
        self.logger = self._setup_logger()
        self.settings = settings
        
        # Validate required fields
        self._validate_required_fields(required_fields)
        
        # Initialize the specific client
        self._initialize_client()
        
        self.logger.info(f"{self.service_name} client initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with proper configuration"""
        # Import secure logger
        try:
            from vigia_detect.utils.secure_logger import get_secure_logger
            logger = get_secure_logger(f'vigia.{self.service_name.lower()}')
            return logger.logger
        except ImportError:
            # Fallback to standard logger
            logger = logging.getLogger(f'vigia.{self.service_name.lower()}')
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(getattr(logging, settings.log_level))
            return logger
    
    def _validate_required_fields(self, required_fields: List[str]):
        """
        Validate that all required fields are present in settings.
        """
        missing_fields = []
        
        for field in required_fields:
            value = getattr(settings, field, None)
            if value is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(
                f"{self.service_name}: Missing required configuration fields: {', '.join(missing_fields)}. "
                f"Please set them in your environment variables."
            )
    
    def get_setting(self, field_name: str, default: Any = None) -> Any:
        """
        Get a setting value with optional default.
        
        Args:
            field_name: Name of the setting field
            default: Default value if field not found
            
        Returns:
            Setting value or default
        """
        return getattr(settings, field_name, default)
    
    @abstractmethod
    def _initialize_client(self):
        """
        Abstract method that each client must implement
        to initialize its specific connection.
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate that the client can connect to the service.
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the client.
        
        Returns:
            Dict with health status information
        """
        try:
            is_valid = self.validate_connection()
            return {
                "service": self.service_name,
                "status": "healthy" if is_valid else "unhealthy",
                "connected": is_valid
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "service": self.service_name,
                "status": "error",
                "error": str(e)
            }