"""
MedGemma Local Client - Cliente para ejecutar MedGemma localmente
Implementación completa para uso de modelos MedGemma descargados desde Hugging Face.

Características:
- Soporte para modelos 4B y 27B
- Inferencia local sin API externa
- Cache de modelos para performance
- Manejo robusto de errores
- Soporte multimodal (texto + imagen)
"""

import asyncio
import json
import logging
import torch
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from PIL import Image
import gc

from ..utils.secure_logger import SecureLogger
from ..utils.error_handling import handle_exceptions
from ..core.base_client_v2 import BaseClientV2

logger = SecureLogger("medgemma_local_client")


class MedGemmaModel(Enum):
    """Modelos MedGemma disponibles."""
    # Modelos oficiales de Google (requieren autenticación)
    MEDGEMMA_4B_IT = "google/medgemma-4b-it"  # 4B instruction-tuned multimodal
    MEDGEMMA_4B_PT = "google/medgemma-4b-pt"  # 4B pre-trained multimodal
    MEDGEMMA_27B_TEXT = "google/medgemma-27b-text-it"  # 27B text-only
    
    # Modelos via Ollama (sin autenticación requerida)
    MEDGEMMA_OLLAMA_27B = "symptoma/medgemma3"  # 27B optimizado por Symptoma
    MEDGEMMA_OLLAMA_4B = "alibayram/medgemma"   # 4B multimodal


class InferenceMode(Enum):
    """Modos de inferencia."""
    TEXT_ONLY = "text_only"
    IMAGE_TEXT = "image_text"
    MULTIMODAL = "multimodal"


@dataclass
class MedGemmaConfig:
    """Configuración para MedGemma local."""
    model_name: MedGemmaModel = MedGemmaModel.MEDGEMMA_4B_IT
    device: str = "auto"  # auto, cpu, cuda
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    quantization: bool = True  # Usar quantización para menor memoria
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    cache_dir: Optional[str] = None
    local_files_only: bool = False  # True para usar solo archivos descargados


@dataclass
class MedicalContext:
    """Contexto médico para análisis."""
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    medical_history: Optional[str] = None
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    symptoms: Optional[str] = None
    urgency_level: str = "routine"  # routine, urgent, emergency


@dataclass
class MedGemmaRequest:
    """Solicitud para MedGemma."""
    text_prompt: str
    image_path: Optional[str] = None
    medical_context: Optional[MedicalContext] = None
    inference_mode: InferenceMode = InferenceMode.TEXT_ONLY
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class MedGemmaResponse:
    """Respuesta de MedGemma."""
    generated_text: str
    confidence_score: float
    processing_time: float
    model_used: str
    inference_mode: str
    medical_analysis: Dict[str, Any]
    warnings: List[str]
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None


class MedGemmaLocalClient(BaseClientV2):
    """Cliente para ejecutar MedGemma localmente."""
    
    def __init__(self, config: Optional[MedGemmaConfig] = None):
        """
        Inicializar cliente MedGemma local.
        
        Args:
            config: Configuración del cliente
        """
        super().__init__()
        self.config = config or MedGemmaConfig()
        
        # Estado del modelo
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        
        # Cache y estadísticas
        self.inference_cache = {}
        self.stats = {
            "requests_processed": 0,
            "total_tokens_generated": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "errors": 0
        }
        
        logger.audit("medgemma_local_client_initialized", {
            "model": self.config.model_name.value,
            "device": self.config.device,
            "quantization": self.config.quantization
        })
    
    async def initialize(self):
        """Inicializar modelo MedGemma."""
        try:
            logger.info(f"Initializing MedGemma model: {self.config.model_name.value}")
            
            # Determinar dispositivo
            self.device = self._determine_device()
            logger.info(f"Using device: {self.device}")
            
            # Cargar modelo en hilo separado para no bloquear
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            
            self.model_loaded = True
            logger.info("MedGemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MedGemma: {e}")
            raise
    
    def _determine_device(self) -> str:
        """Determinar el mejor dispositivo disponible."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                logger.info("Using Apple Silicon MPS")
            else:
                device = "cpu"
                logger.info("Using CPU")
        else:
            device = self.config.device
        
        return device
    
    def _load_model(self):
        """Cargar modelo y tokenizer."""
        try:
            model_name = self.config.model_name.value
            
            # Configurar quantización si está habilitada
            quantization_config = None
            if self.config.quantization and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Determinar torch dtype
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            # Cargar tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only
            )
            
            # Cargar modelo
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only
            )
            
            # Mover a dispositivo si no es CUDA con device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Crear pipeline para inferencia multimodal si es modelo 4B
            if "4b" in model_name:
                try:
                    self.pipeline = pipeline(
                        "image-text-to-text",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch_dtype,
                        device=self.device
                    )
                    logger.info("Multimodal pipeline created")
                except Exception as e:
                    logger.warning(f"Failed to create multimodal pipeline: {e}")
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @handle_exceptions(logger)
    async def generate_medical_response(self, request: MedGemmaRequest) -> MedGemmaResponse:
        """
        Generar respuesta médica usando MedGemma.
        
        Args:
            request: Solicitud con prompt médico
            
        Returns:
            Respuesta generada por MedGemma
        """
        if not self.model_loaded:
            raise RuntimeError("MedGemma model not loaded. Call initialize() first.")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            self.stats["requests_processed"] += 1
            
            # Verificar cache
            cache_key = self._get_cache_key(request)
            if cache_key in self.inference_cache:
                self.stats["cache_hits"] += 1
                cached_response = self.inference_cache[cache_key]
                logger.info("Using cached response")
                return cached_response
            
            # Preparar prompt médico
            formatted_prompt = self._format_medical_prompt(request)
            
            # Generar respuesta
            generated_text = await self._run_inference(request, formatted_prompt)
            
            # Analizar respuesta médica
            medical_analysis = self._analyze_medical_response(generated_text, request)
            
            # Calcular tiempo de procesamiento
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Crear respuesta
            response = MedGemmaResponse(
                generated_text=generated_text,
                confidence_score=medical_analysis.get("confidence", 0.8),
                processing_time=processing_time,
                model_used=self.config.model_name.value,
                inference_mode=request.inference_mode.value,
                medical_analysis=medical_analysis,
                warnings=medical_analysis.get("warnings", []),
                timestamp=datetime.now(timezone.utc),
                success=True
            )
            
            # Guardar en cache
            self.inference_cache[cache_key] = response
            
            # Limpiar cache si es muy grande
            if len(self.inference_cache) > 100:
                oldest_key = next(iter(self.inference_cache))
                del self.inference_cache[oldest_key]
            
            # Actualizar estadísticas
            self._update_stats(response)
            
            logger.audit("medgemma_inference_completed", {
                "processing_time": processing_time,
                "tokens_generated": len(generated_text.split()),
                "inference_mode": request.inference_mode.value,
                "cache_hit": False
            })
            
            return response
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"MedGemma inference failed: {e}")
            
            return MedGemmaResponse(
                generated_text="",
                confidence_score=0.0,
                processing_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                model_used=self.config.model_name.value,
                inference_mode=request.inference_mode.value,
                medical_analysis={},
                warnings=["Error en procesamiento"],
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_message=str(e)
            )
    
    def _format_medical_prompt(self, request: MedGemmaRequest) -> str:
        """Formatear prompt médico para MedGemma."""
        prompt_parts = []
        
        # Contexto médico si está disponible
        if request.medical_context:
            context = request.medical_context
            prompt_parts.append("### Contexto Médico")
            
            if context.patient_age:
                prompt_parts.append(f"Edad del paciente: {context.patient_age} años")
            
            if context.patient_gender:
                prompt_parts.append(f"Género: {context.patient_gender}")
            
            if context.symptoms:
                prompt_parts.append(f"Síntomas: {context.symptoms}")
            
            if context.medical_history:
                prompt_parts.append(f"Historia médica: {context.medical_history}")
            
            if context.current_medications:
                prompt_parts.append(f"Medicamentos actuales: {', '.join(context.current_medications)}")
            
            if context.allergies:
                prompt_parts.append(f"Alergias: {', '.join(context.allergies)}")
            
            prompt_parts.append(f"Nivel de urgencia: {context.urgency_level}")
            prompt_parts.append("")
        
        # Prompt principal
        prompt_parts.append("### Consulta Médica")
        prompt_parts.append(request.text_prompt)
        prompt_parts.append("")
        
        # Instrucciones para el modelo
        prompt_parts.append("### Instrucciones")
        prompt_parts.append("Proporciona una respuesta médica profesional, precisa y basada en evidencia.")
        prompt_parts.append("Incluye recomendaciones específicas y menciona si se necesita evaluación adicional.")
        prompt_parts.append("Si la información es insuficiente, indícalo claramente.")
        prompt_parts.append("")
        prompt_parts.append("### Respuesta:")
        
        return "\n".join(prompt_parts)
    
    async def _run_inference(self, request: MedGemmaRequest, formatted_prompt: str) -> str:
        """Ejecutar inferencia según el modo."""
        
        if request.inference_mode == InferenceMode.IMAGE_TEXT and request.image_path:
            return await self._run_multimodal_inference(request, formatted_prompt)
        else:
            return await self._run_text_inference(formatted_prompt, request)
    
    async def _run_text_inference(self, prompt: str, request: MedGemmaRequest) -> str:
        """Ejecutar inferencia solo texto."""
        loop = asyncio.get_event_loop()
        
        def generate_text():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Parámetros de generación
            generation_config = {
                "max_new_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            # Decodificar solo los tokens nuevos
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
        
        return await loop.run_in_executor(None, generate_text)
    
    async def _run_multimodal_inference(self, request: MedGemmaRequest, formatted_prompt: str) -> str:
        """Ejecutar inferencia multimodal con imagen."""
        if not self.pipeline:
            raise RuntimeError("Multimodal pipeline not available")
        
        if not request.image_path or not os.path.exists(request.image_path):
            raise ValueError("Image path not provided or file doesn't exist")
        
        loop = asyncio.get_event_loop()
        
        def generate_multimodal():
            # Cargar imagen
            image = Image.open(request.image_path).convert("RGB")
            
            # Ejecutar pipeline
            result = self.pipeline(
                image,
                formatted_prompt,
                max_new_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature
            )
            
            return result[0]["generated_text"]
        
        return await loop.run_in_executor(None, generate_multimodal)
    
    def _analyze_medical_response(self, generated_text: str, request: MedGemmaRequest) -> Dict[str, Any]:
        """Analizar respuesta médica para extraer información estructurada."""
        analysis = {
            "confidence": 0.8,  # Valor por defecto
            "warnings": [],
            "medical_entities": [],
            "recommendations": [],
            "urgency_assessment": "routine"
        }
        
        # Detectar palabras de baja confianza
        uncertainty_indicators = [
            "no estoy seguro", "podría ser", "tal vez", "posiblemente",
            "uncertain", "maybe", "possibly", "might be"
        ]
        
        text_lower = generated_text.lower()
        
        for indicator in uncertainty_indicators:
            if indicator in text_lower:
                analysis["confidence"] -= 0.1
                analysis["warnings"].append("Respuesta indica incertidumbre")
                break
        
        # Detectar urgencia
        urgency_keywords = {
            "emergency": ["emergencia", "crítico", "inmediato", "emergency", "critical"],
            "urgent": ["urgente", "pronto", "urgent", "soon"],
            "routine": ["rutina", "regular", "routine", "normal"]
        }
        
        for urgency, keywords in urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis["urgency_assessment"] = urgency
                break
        
        # Extraer recomendaciones (líneas que empiezan con patrones específicos)
        lines = generated_text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if any(line_stripped.startswith(prefix) for prefix in ['- ', '• ', '1.', '2.', '3.']):
                analysis["recommendations"].append(line_stripped)
        
        # Detectar entidades médicas básicas
        medical_terms = [
            "lpp", "lesión por presión", "pressure injury", "ulcer",
            "infección", "infection", "antibiótico", "antibiotic",
            "dolor", "pain", "tratamiento", "treatment"
        ]
        
        for term in medical_terms:
            if term in text_lower:
                analysis["medical_entities"].append(term)
        
        return analysis
    
    def _get_cache_key(self, request: MedGemmaRequest) -> str:
        """Generar clave de cache para la solicitud."""
        import hashlib
        
        cache_data = {
            "prompt": request.text_prompt,
            "image": request.image_path,
            "context": request.medical_context.__dict__ if request.medical_context else None,
            "mode": request.inference_mode.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    def _update_stats(self, response: MedGemmaResponse):
        """Actualizar estadísticas."""
        # Contar tokens aproximadamente
        tokens = len(response.generated_text.split())
        self.stats["total_tokens_generated"] += tokens
        
        # Actualizar tiempo promedio
        total_time = self.stats["average_processing_time"] * (self.stats["requests_processed"] - 1)
        total_time += response.processing_time
        self.stats["average_processing_time"] = total_time / self.stats["requests_processed"]
    
    async def validate_connection(self) -> bool:
        """Validar que el modelo esté cargado y funcional."""
        try:
            if not self.model_loaded:
                return False
            
            # Test simple
            test_request = MedGemmaRequest(
                text_prompt="¿Qué es una lesión por presión?",
                inference_mode=InferenceMode.TEXT_ONLY,
                max_tokens=50
            )
            
            response = await self.generate_medical_response(test_request)
            return response.success and len(response.generated_text) > 0
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cliente."""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.config.model_name.value,
            "device": self.device,
            "requests_processed": self.stats["requests_processed"],
            "total_tokens_generated": self.stats["total_tokens_generated"],
            "average_processing_time": self.stats["average_processing_time"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["requests_processed"], 1),
            "errors": self.stats["errors"],
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Obtener uso de memoria."""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info["cuda_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return memory_info
    
    async def cleanup(self):
        """Limpiar recursos."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.pipeline:
                del self.pipeline
            
            # Limpiar cache de GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Garbage collection
            gc.collect()
            
            self.model_loaded = False
            logger.info("MedGemma client cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class MedGemmaLocalFactory:
    """Factory para crear instancias de MedGemma local."""
    
    @staticmethod
    async def create_client(config: Optional[MedGemmaConfig] = None) -> MedGemmaLocalClient:
        """Crear cliente MedGemma local."""
        client = MedGemmaLocalClient(config)
        await client.initialize()
        return client
    
    @staticmethod
    def check_requirements() -> Dict[str, bool]:
        """Verificar requisitos para MedGemma local."""
        requirements = {
            "torch_available": False,
            "transformers_available": False,
            "cuda_available": False,
            "sufficient_memory": False
        }
        
        try:
            import torch
            requirements["torch_available"] = True
            requirements["cuda_available"] = torch.cuda.is_available()
            
            # Verificar memoria (requiere al menos 8GB para 4B, 32GB para 27B)
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                requirements["sufficient_memory"] = memory_gb >= 8
            
        except ImportError:
            pass
        
        try:
            import transformers
            requirements["transformers_available"] = True
        except ImportError:
            pass
        
        return requirements