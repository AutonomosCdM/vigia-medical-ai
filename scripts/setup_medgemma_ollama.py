#!/usr/bin/env python3
"""
Setup MedGemma con Ollama - Alternativa simple sin autenticación
Instala y configura MedGemma usando Ollama para evitar complejidades de Hugging Face.

Uso:
    python scripts/setup_medgemma_ollama.py --model 27b --install
    python scripts/setup_medgemma_ollama.py --model 4b --test
"""

import os
import sys
import argparse
import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from vigia_detect.utils.secure_logger import SecureLogger

logger = SecureLogger("medgemma_ollama_setup")


class MedGemmaOllamaSetup:
    """Configurador de MedGemma usando Ollama."""
    
    def __init__(self):
        self.models_map = {
            "4b": "alibayram/medgemma",      # 4B multimodal
            "27b": "symptoma/medgemma3"      # 27B text-only optimizado
        }
    
    def check_ollama_installed(self) -> bool:
        """Verificar si Ollama está instalado."""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Ollama instalado: {result.stdout.strip()}")
                return True
            else:
                print("❌ Ollama no responde correctamente")
                return False
        except FileNotFoundError:
            print("❌ Ollama no está instalado")
            return False
    
    def install_ollama(self):
        """Instalar Ollama."""
        print("📦 Instalando Ollama...")
        
        # Detectar sistema operativo
        import platform
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            print("   Instalando en macOS...")
            os.system("curl -fsSL https://ollama.ai/install.sh | sh")
        elif system == "linux":
            print("   Instalando en Linux...")
            os.system("curl -fsSL https://ollama.ai/install.sh | sh")
        else:
            print("❌ Sistema no soportado para instalación automática")
            print("   Instala manualmente desde https://ollama.ai")
            return False
        
        return self.check_ollama_installed()
    
    def list_available_models(self):
        """Listar modelos MedGemma disponibles."""
        print("📋 Modelos MedGemma disponibles via Ollama:")
        print()
        
        for size, model_name in self.models_map.items():
            print(f"🔸 {size}: {model_name}")
            if size == "4b":
                print("   Tipo: Multimodal (texto + imagen)")
                print("   Parámetros: 4B")
                print("   Memoria: ~8GB RAM")
                print("   Ventajas: Más rápido, soporte de imágenes")
            else:
                print("   Tipo: Solo texto")
                print("   Parámetros: 27B") 
                print("   Memoria: ~16GB RAM")
                print("   Ventajas: Más preciso para análisis médico")
            
            # Verificar si está instalado
            if self.check_model_installed(size):
                print("   Estado: ✅ Instalado")
            else:
                print("   Estado: ❌ No instalado")
            print()
    
    def check_model_installed(self, model_size: str) -> bool:
        """Verificar si un modelo está instalado."""
        if model_size not in self.models_map:
            return False
            
        model_name = self.models_map[model_size]
        
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return model_name in result.stdout
            return False
        except Exception:
            return False
    
    def install_model(self, model_size: str) -> bool:
        """Instalar modelo MedGemma."""
        if model_size not in self.models_map:
            print(f"❌ Modelo '{model_size}' no válido. Opciones: {list(self.models_map.keys())}")
            return False
        
        if not self.check_ollama_installed():
            print("❌ Ollama no está instalado. Ejecuta: --install-ollama")
            return False
        
        model_name = self.models_map[model_size]
        print(f"📥 Instalando modelo {model_name}...")
        print(f"   ⏳ Esto puede tomar varios minutos dependiendo de tu conexión...")
        
        try:
            # Ejecutar ollama pull
            result = subprocess.run(["ollama", "pull", model_name], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ Modelo {model_name} instalado exitosamente")
                return True
            else:
                print(f"❌ Error instalando modelo: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando ollama pull: {e}")
            return False
    
    def test_model(self, model_size: str) -> bool:
        """Probar modelo instalado."""
        if not self.check_model_installed(model_size):
            print(f"❌ Modelo {model_size} no está instalado")
            return False
        
        model_name = self.models_map[model_size]
        print(f"🧪 Probando modelo {model_name}...")
        
        # Prompt de prueba médico
        test_prompt = "¿Cuáles son los signos de una lesión por presión grado 2?"
        
        try:
            # Crear comando ollama run
            cmd = ["ollama", "run", model_name, test_prompt]
            
            print(f"   Prompt: {test_prompt}")
            print("   🤖 Respuesta:")
            print("   " + "-" * 50)
            
            # Ejecutar y mostrar respuesta en tiempo real
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                print("   " + "-" * 50)
                print("✅ Modelo funcionando correctamente")
                return True
            else:
                print(f"❌ Error ejecutando modelo: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Error probando modelo: {e}")
            return False
    
    def get_model_info(self, model_size: str) -> Dict[str, Any]:
        """Obtener información del modelo."""
        if not self.check_model_installed(model_size):
            return {"installed": False}
        
        model_name = self.models_map[model_size]
        
        try:
            # Obtener información con ollama show
            result = subprocess.run(["ollama", "show", model_name], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "installed": True,
                    "model_name": model_name,
                    "info": result.stdout
                }
            else:
                return {"installed": False, "error": result.stderr}
                
        except Exception as e:
            return {"installed": False, "error": str(e)}
    
    def get_recommendation(self) -> str:
        """Obtener recomendación de modelo según sistema."""
        import psutil
        
        # Obtener memoria RAM disponible
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"💾 Memoria RAM disponible: {memory_gb:.1f} GB")
        
        if memory_gb >= 32:
            print("💡 Recomendación: Modelo 27b (mejor precisión médica)")
            return "27b"
        elif memory_gb >= 16:
            print("💡 Recomendación: Modelo 27b (con swap si es necesario)")
            return "27b"
        else:
            print("💡 Recomendación: Modelo 4b (más eficiente)")
            return "4b"


async def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Configurar MedGemma con Ollama")
    parser.add_argument("--model", choices=["4b", "27b"], 
                       help="Modelo a instalar/verificar")
    parser.add_argument("--install", action="store_true",
                       help="Instalar modelo")
    parser.add_argument("--test", action="store_true",
                       help="Probar modelo")
    parser.add_argument("--list", action="store_true",
                       help="Listar modelos disponibles")
    parser.add_argument("--install-ollama", action="store_true",
                       help="Instalar Ollama")
    parser.add_argument("--check", action="store_true",
                       help="Verificar instalación")
    
    args = parser.parse_args()
    
    setup = MedGemmaOllamaSetup()
    
    print("🤖 MedGemma Ollama Setup")
    print("=" * 50)
    
    if args.install_ollama:
        setup.install_ollama()
        return
    
    if args.list:
        setup.list_available_models()
        return
    
    if args.check:
        if setup.check_ollama_installed():
            print("✅ Ollama está funcionando correctamente")
        else:
            print("❌ Ollama no está disponible. Ejecuta: --install-ollama")
        return
    
    # Verificar que Ollama esté instalado
    if not setup.check_ollama_installed():
        print("❌ Ollama no está instalado. Ejecuta: --install-ollama")
        return
    
    if not args.model:
        # Mostrar recomendación
        recommended = setup.get_recommendation()
        print(f"💡 Usar: --model {recommended} --install")
        return
    
    if args.install:
        print(f"📥 Instalando modelo {args.model}...")
        success = setup.install_model(args.model)
        if not success:
            return
    
    if args.test:
        print(f"🧪 Probando modelo {args.model}...")
        success = setup.test_model(args.model)
        if success:
            print("✅ Modelo listo para usar con Ollama")
        else:
            print("❌ Modelo no funciona correctamente")


if __name__ == "__main__":
    asyncio.run(main())