#!/usr/bin/env python3
"""
VIGIA Medical AI - Hume AI API Endpoint Test
==========================================

Test the correct Hume AI API endpoint and format for voice analysis.
"""

import asyncio
import os
import sys
import tempfile
import wave
import aiohttp
import numpy as np
from pathlib import Path
import json
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_simple_audio() -> bytes:
    """Generate simple test audio (WAV format)"""
    # Generate 3 seconds of sine wave at 440Hz
    sample_rate = 16000
    duration = 3.0
    frequency = 440.0
    
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    signal = np.sin(2 * np.pi * frequency * t) * 0.8
    
    # Convert to 16-bit PCM
    audio_data = (signal * 32767).astype(np.int16)
    
    # Create WAV file in memory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Read back as bytes
        with open(temp_wav.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Cleanup
        os.unlink(temp_wav.name)
        
        return audio_bytes

async def test_hume_batch_api():
    """Test Hume AI batch API with correct endpoint"""
    api_key = os.getenv("HUME_AI_API_KEY")
    if not api_key:
        print("❌ HUME_AI_API_KEY not found")
        return False
    
    print("🔍 Testing Hume AI Batch API...")
    
    try:
        # Generate test audio
        audio_data = await generate_simple_audio()
        print(f"✅ Generated test audio: {len(audio_data)} bytes")
        
        # Test the correct Hume AI endpoint
        base_url = "https://api.hume.ai/v0"
        
        async with aiohttp.ClientSession() as session:
            # Create form data for file upload
            data = aiohttp.FormData()
            data.add_field('file', audio_data, filename='test_audio.wav', content_type='audio/wav')
            data.add_field('models', 'prosody')  # Voice prosody model
            
            headers = {
                'X-Hume-Api-Key': api_key
            }
            
            # Submit batch job
            print("🚀 Submitting batch job...")
            async with session.post(
                f"{base_url}/batch/jobs",
                data=data,
                headers=headers
            ) as response:
                print(f"📡 Response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    job_id = result.get('job_id')
                    print(f"✅ Job submitted successfully: {job_id}")
                    
                    # Poll for results
                    print("⏳ Polling for results...")
                    for attempt in range(30):  # Max 60 seconds
                        await asyncio.sleep(2)
                        
                        async with session.get(
                            f"{base_url}/batch/jobs/{job_id}",
                            headers=headers
                        ) as status_response:
                            if status_response.status == 200:
                                status_result = await status_response.json()
                                state = status_result.get('state')
                                
                                print(f"   Attempt {attempt + 1}: {state}")
                                
                                if state == 'COMPLETED':
                                    # Get predictions
                                    async with session.get(
                                        f"{base_url}/batch/jobs/{job_id}/predictions",
                                        headers=headers
                                    ) as pred_response:
                                        if pred_response.status == 200:
                                            predictions = await pred_response.json()
                                            print("✅ Predictions received!")
                                            
                                            # Extract emotions from prosody
                                            if 'predictions' in predictions and predictions['predictions']:
                                                pred = predictions['predictions'][0]
                                                models = pred.get('models', {})
                                                prosody = models.get('prosody', {})
                                                
                                                if 'grouped_predictions' in prosody:
                                                    grouped = prosody['grouped_predictions'][0]
                                                    emotions = grouped['predictions'][0]['emotions']
                                                    
                                                    print("🎭 Top emotions detected:")
                                                    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
                                                    for emotion in sorted_emotions[:10]:
                                                        print(f"   {emotion['name']}: {emotion['score']:.3f}")
                                                    
                                                    # Check for medical-relevant emotions
                                                    medical_emotions = {}
                                                    for emotion in emotions:
                                                        name = emotion['name'].lower()
                                                        if 'pain' in name or 'distress' in name or 'fear' in name:
                                                            medical_emotions[emotion['name']] = emotion['score']
                                                    
                                                    if medical_emotions:
                                                        print("🩺 Medical-relevant emotions:")
                                                        for name, score in medical_emotions.items():
                                                            print(f"   {name}: {score:.3f}")
                                                    else:
                                                        print("ℹ️ No direct medical emotions detected in this test sample")
                                            
                                            return True
                                
                                elif state == 'FAILED':
                                    error_msg = status_result.get('message', 'Unknown error')
                                    print(f"❌ Job failed: {error_msg}")
                                    return False
                    
                    print("⏰ Polling timeout")
                    return False
                    
                elif response.status == 401:
                    print("❌ Authentication failed - check API key")
                    return False
                elif response.status == 400:
                    error_text = await response.text()
                    print(f"❌ Bad request: {error_text}")
                    return False
                else:
                    error_text = await response.text()
                    print(f"❌ API error {response.status}: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_hume_websocket():
    """Test WebSocket streaming (basic connection test)"""
    print("\\n🔌 Testing Hume AI WebSocket connection...")
    
    try:
        import websockets
        
        api_key = os.getenv("HUME_AI_API_KEY")
        ws_url = f"wss://api.hume.ai/v0/expression/stream/models?apikey={api_key}"
        
        # Basic connection test
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connection established")
            
            # Send config message
            config = {
                "models": {
                    "prosody": {}
                }
            }
            
            await websocket.send(json.dumps(config))
            response = await websocket.recv()
            print(f"✅ WebSocket configured: {json.loads(response)}")
            
            return True
            
    except ImportError:
        print("⚠️ websockets library not installed - skipping WebSocket test")
        return True
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

async def main():
    """Run Hume AI endpoint tests"""
    print("🩺 VIGIA Medical AI - Hume AI Endpoint Testing")
    print("=" * 55)
    
    # Check API key
    api_key = os.getenv("HUME_AI_API_KEY")
    if not api_key:
        print("❌ HUME_AI_API_KEY not found in environment")
        print("   Please set the API key in .env file")
        return False
    
    print(f"🔑 API Key configured: {api_key[:20]}...")
    
    # Test batch API
    batch_success = await test_hume_batch_api()
    
    # Test WebSocket (optional)
    ws_success = await test_hume_websocket()
    
    print("\\n" + "=" * 55)
    if batch_success:
        print("🎉 Hume AI batch API is working correctly!")
        print("✅ Ready for integration with VIGIA Medical AI")
    else:
        print("❌ Hume AI API connection failed")
        print("   Check API key and network connectivity")
    
    print(f"📊 Batch API: {'✅' if batch_success else '❌'}")
    print(f"📊 WebSocket: {'✅' if ws_success else '❌'}")
    
    return batch_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)