"""
Clase base para todos los clientes de servicios externos
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class BaseClient(ABC):
    """
    Clase base abstracta para clientes de servicios externos.
    Maneja la inicialización común, carga de credenciales y logging.
    """
    
    def __init__(self, service_name: str, required_env_vars: Dict[str, str], 
                 optional_env_vars: Optional[Dict[str, str]] = None):
        """
        Inicializa el cliente base.
        
        Args:
            service_name: Nombre del servicio (para logging)
            required_env_vars: Dict de {attr_name: ENV_VAR_NAME} requeridas
            optional_env_vars: Dict de {attr_name: ENV_VAR_NAME} opcionales
        """
        self.service_name = service_name
        self.logger = self._setup_logger()
        
        # Cargar variables requeridas
        self._load_required_env_vars(required_env_vars)
        
        # Cargar variables opcionales
        if optional_env_vars:
            self._load_optional_env_vars(optional_env_vars)
        
        # Inicializar el cliente específico
        self._initialize_client()
        
        self.logger.info(f"{self.service_name} client initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura el logger para el cliente con seguridad mejorada"""
        # Import secure logger
        try:
            from vigia_detect.utils.secure_logger import get_secure_logger
            logger = get_secure_logger(f'vigia.{self.service_name.lower()}')
            return logger.logger
        except ImportError:
            # Fallback to standard logger if secure logger not available
            logger = logging.getLogger(f'vigia.{self.service_name.lower()}')
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_required_env_vars(self, env_vars: Dict[str, str]):
        """
        Carga variables de entorno requeridas.
        Lanza excepción si alguna falta.
        """
        missing_vars = []
        
        for attr_name, env_var_name in env_vars.items():
            value = os.getenv(env_var_name)
            if value:
                setattr(self, attr_name, value)
            else:
                missing_vars.append(env_var_name)
        
        if missing_vars:
            raise ValueError(
                f"{self.service_name}: Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please set them in your .env file."
            )
    
    def _load_optional_env_vars(self, env_vars: Dict[str, str]):
        """
        Carga variables de entorno opcionales.
        Usa None como valor por defecto si no están definidas.
        """
        for attr_name, env_var_name in env_vars.items():
            value = os.getenv(env_var_name)
            setattr(self, attr_name, value)
            if not value:
                self.logger.debug(f"Optional variable {env_var_name} not set")
    
    @abstractmethod
    def _initialize_client(self):
        """
        Método abstracto que cada cliente debe implementar
        para inicializar su conexión específica.
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Verifica que el cliente esté funcionando correctamente.
        Retorna True si está saludable, False en caso contrario.
        """
        pass
    
    def log_error(self, operation: str, error: Exception):
        """Log de errores estandarizado"""
        self.logger.error(f"Error in {operation}: {str(error)}", exc_info=True)
    
    def log_info(self, message: str):
        """Log de información"""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log de debug"""
        self.logger.debug(message)