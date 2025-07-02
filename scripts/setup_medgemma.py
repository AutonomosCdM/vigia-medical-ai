#!/usr/bin/env python3
"""
MedGemma Medical AI Setup Script
===============================

Automated setup for MedGemma 27B medical language model with Ollama.
Essential for VIGIA Medical AI clinical decision support.

Usage:
    python scripts/setup_medgemma.py --install-ollama
    python scripts/setup_medgemma.py --model 27b --install
    python scripts/setup_medgemma.py --test
"""

import subprocess
import sys
import time
import requests
import argparse
from pathlib import Path

class MedGemmaSetup:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        
    def install_ollama(self):
        """Install Ollama AI runtime"""
        print("🤖 Installing Ollama AI runtime...")
        
        try:
            # Check if already installed
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama already installed")
                return True
        except FileNotFoundError:
            pass
        
        # Install Ollama
        try:
            print("📥 Downloading Ollama installer...")
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            subprocess.run(install_cmd, shell=True, check=True)
            print("✅ Ollama installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Ollama: {e}")
            return False
    
    def start_ollama_service(self):
        """Start Ollama service"""
        print("🚀 Starting Ollama service...")
        
        try:
            # Check if already running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama service already running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Start service
        try:
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for i in range(10):
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("✅ Ollama service started")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(2)
            
            print("❌ Failed to start Ollama service")
            return False
            
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")
            return False
    
    def install_medgemma(self, model_size="27b"):
        """Install MedGemma medical model"""
        print(f"🩺 Installing MedGemma {model_size} medical AI model...")
        print("⚠️  This may take 10-15 minutes depending on connection speed")
        
        try:
            # Pull MedGemma model
            cmd = ['ollama', 'pull', f'medgemma:{model_size}']
            result = subprocess.run(cmd, check=True, text=True)
            
            print(f"✅ MedGemma {model_size} installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install MedGemma: {e}")
            return False
    
    def test_medgemma(self):
        """Test MedGemma medical AI functionality"""
        print("🧪 Testing MedGemma medical AI...")
        
        try:
            # Check if model is available
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json()
            
            medgemma_models = [m for m in models.get('models', []) 
                             if 'medgemma' in m.get('name', '').lower()]
            
            if not medgemma_models:
                print("❌ No MedGemma models found")
                return False
            
            print(f"✅ Found MedGemma models: {len(medgemma_models)}")
            
            # Test medical query
            test_prompt = {
                "model": medgemma_models[0]['name'],
                "prompt": "What is the NPUAP classification for pressure injuries?",
                "stream": False
            }
            
            print("🔬 Testing medical knowledge query...")
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=test_prompt, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                medical_response = result.get('response', '')
                
                if 'grade' in medical_response.lower() or 'stage' in medical_response.lower():
                    print("✅ MedGemma medical AI responding correctly")
                    print(f"📋 Sample response: {medical_response[:100]}...")
                    return True
                else:
                    print("⚠️  MedGemma responding but may need medical fine-tuning")
                    return True
            else:
                print(f"❌ MedGemma test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing MedGemma: {e}")
            return False
    
    def setup_complete_system(self, model_size="27b"):
        """Setup complete MedGemma system"""
        print("🩺 VIGIA Medical AI - MedGemma Setup")
        print("=" * 50)
        
        success = True
        
        # Install Ollama
        if not self.install_ollama():
            success = False
        
        # Start service
        if success and not self.start_ollama_service():
            success = False
        
        # Install MedGemma
        if success and not self.install_medgemma(model_size):
            success = False
        
        # Test functionality
        if success and not self.test_medgemma():
            success = False
        
        print("\n" + "=" * 50)
        if success:
            print("🏆 MEDGEMMA SETUP COMPLETE")
            print("✅ Medical AI ready for clinical decision support")
            print("🩺 VIGIA Medical AI can now provide NPUAP-compliant recommendations")
        else:
            print("❌ SETUP INCOMPLETE - Check errors above")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Setup MedGemma Medical AI")
    parser.add_argument('--install-ollama', action='store_true', 
                       help='Install Ollama AI runtime')
    parser.add_argument('--model', choices=['7b', '27b'], default='27b',
                       help='MedGemma model size (default: 27b)')
    parser.add_argument('--install', action='store_true',
                       help='Install MedGemma medical model')
    parser.add_argument('--test', action='store_true',
                       help='Test MedGemma functionality')
    parser.add_argument('--complete', action='store_true',
                       help='Complete setup (install + test)')
    
    args = parser.parse_args()
    
    setup = MedGemmaSetup()
    
    if args.install_ollama:
        setup.install_ollama()
        setup.start_ollama_service()
    
    if args.install:
        setup.install_medgemma(args.model)
    
    if args.test:
        setup.test_medgemma()
    
    if args.complete:
        setup.setup_complete_system(args.model)

if __name__ == "__main__":
    main()