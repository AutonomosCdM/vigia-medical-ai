#!/usr/bin/env python3
"""
Simple Hume AI API Test
"""

import asyncio
import os
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

async def test_simple_request():
    """Test simple API request to validate key"""
    api_key = os.getenv("HUME_AI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"🔑 Testing API key: {api_key[:20]}...")
    
    # Test basic authentication with a simple endpoint
    async with aiohttp.ClientSession() as session:
        headers = {
            'X-Hume-Api-Key': api_key,
            'Accept': 'application/json'
        }
        
        # Try to list jobs (should work even if empty)
        try:
            async with session.get(
                "https://api.hume.ai/v0/batch/jobs",
                headers=headers
            ) as response:
                print(f"📡 Status: {response.status}")
                text = await response.text()
                print(f"📄 Response: {text[:200]}")
                
                if response.status == 200:
                    print("✅ API key is valid!")
                    return True
                elif response.status == 401:
                    print("❌ API key is invalid or expired")
                    return False
                else:
                    print(f"❓ Unexpected status: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_request())
    print(f"Result: {'✅' if success else '❌'}")